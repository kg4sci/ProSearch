import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import json_repair
import openai

from prompt import *


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_existing_ids(path):
    if not os.path.exists(path):
        return set()

    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("candidate_id")
                if cid:
                    done.add(cid)
            except Exception:
                pass
    return done


def load_keys(path):
    with open(path, "r", encoding="utf-8") as f:
        keys = json.load(f)
    return keys



def pretty_json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)


def extract_first_json_object(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        try:
            return json_repair.loads(text)
        except Exception:
            pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find JSON object in response: {text[:500]}")

    matched_text = match.group(0)
    try:
        return json.loads(matched_text)
    except Exception:
        try:
            return json_repair.loads(matched_text)
        except Exception as e:
            raise ValueError(f"Failed to parse extracted JSON object: {matched_text[:500]}") from e


def build_prompt_record(rec):
    prompt_rec = {
        "type": rec["type"],
        "candidate_id": rec["candidate_id"],
        "constraints": rec["constraints"],
    }
    if rec["type"] != "reproduction":
        prompt_rec["explore_field"] = rec.get("explore_field")
    return prompt_rec


def get_relevant_field_defs(domain, rec, field_defs):
    used_fields = []

    for c in rec.get("constraints", []):
        f = c.get("field")
        if f:
            used_fields.append(f)

    explore_field = rec.get("explore_field")
    if explore_field:
        used_fields.append(explore_field)

    seen = set()
    used_fields = [f for f in used_fields if not (f in seen or seen.add(f))]

    domain_defs = field_defs[domain]
    relevant_defs = {f: domain_defs[f] for f in used_fields if f in domain_defs}
    return relevant_defs


def choose_gen_prompt(rec, domain):
    relevant_field_defs = get_relevant_field_defs(domain, rec, FIELD_DEFS)
    field_defs_pretty = pretty_json(relevant_field_defs)
    record_pretty_json = pretty_json(build_prompt_record(rec))

    if rec["type"] == "reproduction":
        user_prompt = gen_prompt_reproduction.format(
            domain=domain,
            field_definitions_pretty=field_defs_pretty,
            record_pretty_json=record_pretty_json,
        )
    else:
        user_prompt = gen_prompt_exploration.format(
            domain=domain,
            field_definitions_pretty=field_defs_pretty,
            record_pretty_json=record_pretty_json,
        )

    system_prompt = sys_prompt.format(domain=domain)
    return system_prompt, user_prompt


def choose_eval_prompt(rec, domain):
    relevant_field_defs = get_relevant_field_defs(domain, rec, FIELD_DEFS)
    field_defs_pretty = pretty_json(relevant_field_defs)
    record_pretty_json = pretty_json(build_prompt_record(rec))
    generated_query = rec["request_query"]

    user_prompt = eval_prompt.format(
        domain=domain,
        field_definitions_pretty=field_defs_pretty,
        record_pretty_json=record_pretty_json,
        generated_query=generated_query,
    )

    system_prompt = sys_prompt.format(domain=domain)
    return system_prompt, user_prompt


def validate_input_record(rec, expected_domain):
    if rec.get("domain") != expected_domain:
        raise ValueError(
            f"Domain mismatch: file record domain={rec.get('domain')} "
            f"but --domain={expected_domain}"
        )

    if rec.get("type") not in {
        "reproduction",
        "input_exploration",
        "condition_exploration",
        "method_exploration",
    }:
        raise ValueError(f"Unknown type: {rec.get('type')}")

    if "candidate_id" not in rec:
        raise ValueError("Missing candidate_id")

    if "constraints" not in rec:
        raise ValueError("Missing constraints")

    if rec["type"] != "reproduction" and not rec.get("explore_field"):
        raise ValueError("Exploration sample missing explore_field")


def call_model(entry, domain, task_type, worker_id, api_keys, args):
    if task_type == "gen":
        system_prompt_text, user_prompt = choose_gen_prompt(entry, domain)
        temperature = args.gen_temperature
    elif task_type == "eval":
        system_prompt_text, user_prompt = choose_eval_prompt(entry, domain)
        temperature = args.eval_temperature
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    last_error = None

    for attempt in range(args.max_retries):
        api_key = api_keys[(worker_id + attempt) % len(api_keys)]

        try:
            client = openai.OpenAI(
                base_url=args.base_url,
                api_key=api_key,
            )

            response = client.chat.completions.create(
                model=args.model,
                temperature=temperature,
                max_tokens=args.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_text = response.choices[0].message.content
            return raw_text

        except Exception as e:
            last_error = e
            print(
                f"[Retry {attempt + 1}/{args.max_retries}] "
                f"Failed on candidate_id={entry.get('candidate_id')} "
                f"with worker={worker_id}: {e}"
            )
            if attempt < args.max_retries - 1:
                time.sleep(args.sleep_seconds * (attempt + 1))

    raise RuntimeError(f"Failed after {args.max_retries} retries: {last_error}")


def process_gen_record(rec, args, worker_id, api_keys):
    validate_input_record(rec, args.domain)

    raw = call_model(
        entry=rec,
        domain=args.domain,
        task_type="gen",
        worker_id=worker_id,
        api_keys=api_keys,
        args=args,
    )
    obj = extract_first_json_object(raw)

    result = {
        "domain": rec["domain"],
        "type": rec["type"],
        "type_id": rec.get("type_id"),
        "candidate_id": rec["candidate_id"],
        "constraints": rec["constraints"],
        "explore_field": rec.get("explore_field"),
        "gen_model": args.model,
        "request_query": obj.get("request_query"),
        "skip": obj.get("skip"),
        "gen_raw_response": raw,
    }
    return result


def process_eval_record(rec, args, worker_id, api_keys):
    validate_input_record(rec, args.domain)

    if rec.get("skip") is True or not rec.get("request_query"):
        return {
            "domain": rec["domain"],
            "type": rec["type"],
            "type_id": rec.get("type_id"),
            "candidate_id": rec["candidate_id"],
            "constraints": rec["constraints"],
            "explore_field": rec.get("explore_field"),
            "request_query": rec.get("request_query"),
            "skip": rec.get("skip"),
            "eval_model": args.model,
            "eval_status": "skipped",
        }

    raw = call_model(
        entry=rec,
        domain=args.domain,
        task_type="eval",
        worker_id=worker_id,
        api_keys=api_keys,
        args=args,
    )
    obj = extract_first_json_object(raw)

    result = {
        "domain": rec["domain"],
        "type": rec["type"],
        "type_id": rec.get("type_id"),
        "candidate_id": rec["candidate_id"],
        "constraints": rec["constraints"],
        "explore_field": rec.get("explore_field"),
        "request_query": rec.get("request_query"),
        "skip": rec.get("skip"),
        "eval_model": args.model,
        "eval_raw_response": raw,
    }
    result.update(obj)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task_type", choices=["gen", "eval"], required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--keys_path", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--mode", choices=["overwrite", "supplement"], default="supplement")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=2)
    parser.add_argument("--gen_temperature", type=float, default=0.7)
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=40)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    records = load_jsonl(args.input_path)
    # records = records[:5]

    done_ids = load_existing_ids(args.output_path) if args.mode == "supplement" else set()
    todo = [r for r in records if r.get("candidate_id") not in done_ids]

    if args.mode == "overwrite" and os.path.exists(args.output_path):
        os.remove(args.output_path)

    print(f"Loaded {len(records)} records; {len(todo)} to process.")

    api_keys = load_keys(args.keys_path)
    # print(api_keys[0])
    fn = process_gen_record if args.task_type == "gen" else process_eval_record

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {}
        for i, rec in enumerate(todo):
            worker_id = i
            futures[ex.submit(fn, rec, args, worker_id, api_keys)] = rec

        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                out = fut.result()
                append_jsonl(args.output_path, out)
                print(f"[OK] {rec['candidate_id']}")
            except Exception as e:
                print(f"[ERR] {rec.get('candidate_id')}: {e}")


if __name__ == "__main__":
    main()
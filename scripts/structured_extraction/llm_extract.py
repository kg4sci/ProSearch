import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm
import json_repair

from llm_extract_prompt import render_extract_prompt


_thread_local = threading.local()
_write_lock = threading.Lock()


def load_api_keys(path: str):
    with open(path, "r", encoding="utf-8") as f:
        keys = json.load(f)
    if not isinstance(keys, list) or not keys:
        raise ValueError("keys.json must be a non-empty JSON list, e.g. ['key1', 'key2']")
    return [k.strip() for k in keys if isinstance(k, str) and k.strip()]



def load_processed_ids(out_path: str):
    processed = set()
    p = Path(out_path)
    if not p.exists():
        return processed

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("result") is not None and obj.get("id") is not None:
                processed.add(str(obj["id"]))
    return processed



def get_client(base_url: str, api_key: str, timeout: int):
    if not hasattr(_thread_local, "clients"):
        _thread_local.clients = {}

    key = (base_url, api_key, timeout)
    if key not in _thread_local.clients:
        _thread_local.clients[key] = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
    return _thread_local.clients[key]



def append_jsonl(out_path: str, record: dict):
    with _write_lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



def call_model(item, domain, slot_id, base_url, model_name, api_keys, max_tokens, temperature, timeout, max_retries, sleep_seconds):
    system_prompt, user_prompt = render_extract_prompt(domain, item["paragraph"])
    last_error = None

    for attempt in range(max_retries):
        api_key = api_keys[(slot_id + attempt) % len(api_keys)]
        try:
            client = get_client(base_url, api_key, timeout)
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = json_repair.loads(content)
            return parsed, content
        except Exception as e:
            last_error = e
            print(
                f"[Retry {attempt + 1}/{max_retries}] "
                f"domain={domain} id={item.get('id')} slot={slot_id} error={e}"
            )
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds * (attempt + 1))

    raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--keys-json", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--mode", choices=["overwrite", "supplement"], default="supplement")
    args = parser.parse_args()

    api_keys = load_api_keys(args.keys_json)

    with open(args.input, "r", encoding="utf-8") as f:
        all_items = json.load(f)
    # all_items = all_items[:50000]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "overwrite":
        processed_ids = set()
        if Path(args.output).exists():
            print(f"Mode is 'overwrite', clearing existing file: {args.output}")
            open(args.output, "w", encoding="utf-8").close()
    else:
        processed_ids = load_processed_ids(args.output)

    items = [x for x in all_items if str(x.get("id")) not in processed_ids]
    
    # test_id ={
    #     "12222": 1,
    #     # "14435": 1,
    #     # "17542": 1,
    #     # "33680": 1,
    #     # "34271": 1,
    #     "36142": 1,
    #     # "37493": 1,
    #     "3899": 1,
    #     # "45944": 1,
    #     # "5508": 1
    # }
    # items = [x for x in all_items if str(x.get("id")) in test_id]

    print(f"domain={args.domain}")
    print(f"total_input={len(all_items)}")
    print(f"already_done={len(processed_ids)}")
    print(f"remaining={len(items)}")

    if not items:
        print("Nothing to do.")
        return

    workers = min(args.workers, len(items))
    success = 0
    error = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {}
        for i, item in enumerate(items):
            slot_id = i % max(1, len(api_keys))
            future = executor.submit(
                call_model,
                item,
                args.domain,
                slot_id,
                args.base_url,
                args.model,
                api_keys,
                args.max_tokens,
                args.temperature,
                args.timeout,
                args.max_retries,
                args.sleep_seconds,
            )
            future_to_id[future] = item.get("id")

        with tqdm(total=len(future_to_id), desc=args.domain, ncols=100) as pbar:
            for future in as_completed(future_to_id):
                record_id = future_to_id[future]
                try:
                    result, raw = future.result()
                    append_jsonl(
                        args.output,
                        {
                            "domain": args.domain,
                            "id": record_id,
                            "result": result,
                        },
                    )
                    success += 1
                except Exception as e:
                    error += 1
                pbar.update(1)
                pbar.set_postfix(success=success, error=error)

    print(f"done: {args.output}")
    print(f"success={success}, error={error}")


if __name__ == "__main__":
    main()

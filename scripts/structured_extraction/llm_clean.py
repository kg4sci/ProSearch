import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple, Optional

import json_repair
from openai import OpenAI
from tqdm import tqdm

from llm_clean_prompt import render_prompt, FIELD_REQUIREMENTS


_thread_local = threading.local()


def load_api_keys(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        keys = json.load(f)
    if not isinstance(keys, list) or not keys:
        raise ValueError("keys.json must be a non-empty JSON list.")
    return [str(k).strip() for k in keys if str(k).strip()]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object: {slot_name: [...items...]}")
    return data


def atomic_write_json(path: str, data: dict):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(out_path.parent)) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name

    Path(tmp_path).replace(out_path)


def normalize_string_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    cleaned: List[str] = []
    seen = set()
    for x in value:
        if not isinstance(x, str):
            continue
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        cleaned.append(x)
    return cleaned


def normalize_input_item(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None

    original_value = item.get("original_value")
    evidence = item.get("evidence", "")
    value = normalize_string_list(item.get("value"))
    previous_value = normalize_string_list(item.get("previous_value"))

    if not isinstance(original_value, str):
        return None
    original_value = original_value.strip()
    if not original_value:
        return None

    if not value:
        value = [original_value]
    if not previous_value:
        previous_value = value.copy()

    return {
        "original_value": original_value,
        "value": value,
        "evidence": evidence,
        "previous_value": previous_value,
    }


def normalize_slot_items(items: list) -> List[dict]:
    normalized: List[dict] = []
    for item in items:
        norm = normalize_input_item(item)
        if norm is not None:
            normalized.append(norm)
    return normalized


def chunk_list(items: List[dict], batch_size: int) -> List[List[dict]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def make_output_item(original_value: str, value: List[str], evidence: str, previous_value: List[str]) -> dict:
    return {
        "original_value": original_value,
        "value": value,
        "evidence": evidence,
        "previous_value": previous_value,
    }


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


def prepare_batches(
    domain: str,
    input_data: dict,
    batch_size: int,
    test_slot: Optional[str] = None,
    max_batches_per_slot: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    if domain not in FIELD_REQUIREMENTS:
        raise ValueError(f"Domain not found in FIELD_REQUIREMENTS: {domain}")

    allowed_slots = FIELD_REQUIREMENTS[domain]
    normalized_inputs: Dict[str, List[dict]] = {}
    batches: List[dict] = []

    for slot_name, raw_items in input_data.items():
        if test_slot is not None and slot_name != test_slot:
            continue

        if not isinstance(raw_items, list):
            continue

        slot_items = normalize_slot_items(raw_items)

        if slot_name not in allowed_slots or not slot_items:
            continue

        slot_item_batches = chunk_list(slot_items, batch_size)
        if max_batches_per_slot is not None:
            slot_item_batches = slot_item_batches[:max_batches_per_slot]

        normalized_inputs[slot_name] = [
            item for batch in slot_item_batches for item in batch
        ]

        for batch_index, batch_items in enumerate(slot_item_batches):
            prompt_items = []
            for item_index, item in enumerate(batch_items):
                prompt_items.append(
                    {
                        "item_id": f"item_{item_index}",
                        "current_values": item["value"],
                        "evidence": item["evidence"],
                    }
                )

            batches.append(
                {
                    "domain": domain,
                    "slot_name": slot_name,
                    "batch_index": batch_index,
                    "slot_items": batch_items,
                    "prompt_items": prompt_items,
                }
            )

    return batches, normalized_inputs

def validate_model_result(prompt_items: List[dict], parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError(f"Model output must be a JSON object, got: {type(parsed)}")

    expected_keys = [x["item_id"] for x in prompt_items]
    expected_set = set(expected_keys)
    parsed_set = set(parsed.keys())

    missing = expected_set - parsed_set
    extra = parsed_set - expected_set
    if missing or extra:
        raise ValueError(
            f"Model output keys mismatch. missing={sorted(missing)} extra={sorted(extra)}"
        )

    normalized = {}
    for key in expected_keys:
        normalized[key] = normalize_string_list(parsed[key])
    return normalized


def call_model(
    batch_item: dict,
    base_url: str,
    model_name: str,
    api_keys: List[str],
    max_tokens: int,
    temperature: float,
    timeout: int,
    max_retries: int,
    sleep_seconds: float,
    worker_slot: int,
) -> Tuple[dict, str]:
    system_prompt, user_prompt = render_prompt(
        domain=batch_item["domain"],
        slot_name=batch_item["slot_name"],
        batch_items=batch_item["prompt_items"],
    )

    last_error = None
    for attempt in range(max_retries):
        api_key = api_keys[(worker_slot + attempt) % len(api_keys)]
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
            validated = validate_model_result(batch_item["prompt_items"], parsed)
            return validated, content
        except Exception as e:
            last_error = e
            print(
                f"[Retry {attempt + 1}/{max_retries}] "
                f"domain={batch_item['domain']} slot={batch_item['slot_name']} "
                f"batch={batch_item['batch_index']} error={e}"
            )
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds * (attempt + 1))

    raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")


def apply_batch_result(batch_item: dict, normalized_result: dict) -> List[dict]:
    output_items: List[dict] = []
    for item_index, source_item in enumerate(batch_item["slot_items"]):
        item_id = f"item_{item_index}"
        output_items.append(
            make_output_item(
                original_value=source_item["original_value"],
                value=normalized_result[item_id],
                evidence=source_item["evidence"],
                previous_value=source_item["value"],
            )
        )
    return output_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--input", required=True, help="input JSON in stage-compatible format")
    parser.add_argument("--output", required=True, help="output JSON in the same stage-compatible format")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--keys-json", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--test", default=False)
    parser.add_argument("--test-slot", default=None)
    parser.add_argument("--max-batches-per-slot", type=int, default=None)
    args = parser.parse_args()

    api_keys = load_api_keys(args.keys_json)
    input_data = load_json(args.input)

    if args.test == 'true':
        batches, normalized_inputs = prepare_batches(
            domain=args.domain,
            input_data=input_data,
            batch_size=args.batch_size,
            test_slot=args.test_slot,
            max_batches_per_slot=args.max_batches_per_slot,
        )
    elif args.test == 'false':
        batches, normalized_inputs = prepare_batches(
            domain=args.domain,
            input_data=input_data,
            batch_size=args.batch_size,
            test_slot=None,
            max_batches_per_slot=None,
        )

    print(f"domain={args.domain}")
    print(f"prepared_batches={len(batches)}")
    print(f"workers={min(args.workers, max(1, len(batches)))}")

    if not batches:
        print(f'test={args.test} test_slot={args.test_slot}')
        atomic_write_json(args.output, normalized_inputs)
        print("Nothing to do.")
        print(f"done_json={args.output}")
        return

    results_by_slot: Dict[str, Dict[int, List[dict]]] = {}
    success = 0
    error = 0

    workers = min(args.workers, len(batches))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_batch = {}
        for i, batch_item in enumerate(batches):
            worker_slot = i % max(1, len(api_keys))
            future = executor.submit(
                call_model,
                batch_item,
                args.base_url,
                args.model,
                api_keys,
                args.max_tokens,
                args.temperature,
                args.timeout,
                args.max_retries,
                args.sleep_seconds,
                worker_slot,
            )
            future_to_batch[future] = batch_item

        with tqdm(total=len(future_to_batch), desc=f"{args.domain}-normalize", ncols=120) as pbar:
            for future in as_completed(future_to_batch):
                batch_item = future_to_batch[future]
                try:
                    result, _raw = future.result()
                    batch_output_items = apply_batch_result(batch_item, result)
                    results_by_slot.setdefault(batch_item["slot_name"], {})[batch_item["batch_index"]] = batch_output_items
                    success += 1
                except Exception as e:
                    error += 1
                    print(
                        f"[ERROR] slot={batch_item['slot_name']} batch={batch_item['batch_index']} error={e}"
                    )
                pbar.update(1)
                pbar.set_postfix(success=success, error=error)

    if error > 0:
        raise RuntimeError(f"There were {error} failed batches. Output was not written.")

    final_output: Dict[str, List[dict]] = {}
    for slot_name, items in normalized_inputs.items():
        if slot_name not in FIELD_REQUIREMENTS.get(args.domain, {}):
            final_output[slot_name] = items
            continue

        slot_batches = results_by_slot.get(slot_name, {})
        rebuilt_items: List[dict] = []
        for batch_index in range(len(chunk_list(items, args.batch_size))):
            rebuilt_items.extend(slot_batches[batch_index])
        final_output[slot_name] = rebuilt_items

    atomic_write_json(args.output, final_output)
    print(f"done_json={args.output}")
    print(f"success={success}, error={error}")


if __name__ == "__main__":
    main()

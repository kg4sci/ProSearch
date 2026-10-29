import os
import json
import time
import asyncio
import itertools
from typing import Dict, List, Optional

import httpx
import json_repair

from scripts.structured_extraction.llm_clean_prompt import render_prompt, FIELD_DEFS


ENDPOINTS = [
   "http://localhost:8002/v1/chat/completions",
    "http://localhost:8004/v1/chat/completions",
]

MODEL_NAME = "/home/liangwenliang/Model/Qwen3-8B"



def load_field_values(input_path: str) -> Dict[str, List[str]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for field, values in data.items():
        if not isinstance(values, list):
            continue

        uniq = []
        seen = set()
        for v in values:
            if not isinstance(v, str):
                continue
            vv = v.strip()
            if not vv:
                continue
            if vv not in seen:
                seen.add(vv)
                uniq.append(vv)
        out[field] = uniq
    return out


def get_val_batches(values: List[str], batch_size: int = 40) -> List[List[str]]:
    return [values[i:i + batch_size] for i in range(0, len(values), batch_size)]


def build_request(prompt: str, max_tokens: int = 1024):
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


async def one_call(client: httpx.AsyncClient, url: str, payload: dict, timeout=180):
    r = await client.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def normalize_list_value(v) -> Optional[List[str]]:
    if not isinstance(v, list):
        return None

    cleaned = []
    seen = set()
    for x in v:
        if not isinstance(x, str):
            return None
        xx = x.strip()
        if not xx:
            continue
        if xx not in seen:
            seen.add(xx)
            cleaned.append(xx)

    cleaned = sorted(cleaned)
    return cleaned


def safe_parse_mapping(content: str, raw_values: List[str]):
    try:
        output = json_repair.loads(content)
    except Exception:
        return None, "json_parse_failed"

    if not isinstance(output, dict):
        return None, "output_not_dict"

    if set(output.keys()) != set(raw_values):
        return None, "key_mismatch"

    fixed = {}
    for k in raw_values:
        vv = normalize_list_value(output[k])
        if vv is None:
            return None, "non_list_value"
        fixed[k] = vv

    return fixed, None


async def run_all(items, out_merged_path: str, concurrency: int = 32):
    rr = itertools.cycle(ENDPOINTS)
    sem = asyncio.Semaphore(concurrency)
    merged = {}

    async with httpx.AsyncClient() as client:
        async def worker(item):
            async with sem:
                payload = build_request(item["prompt"])
                url = next(rr)

                content = await one_call(client, url, payload)
                output, err = safe_parse_mapping(content, item["values"])

                if err is not None:
                    content = await one_call(client, url, payload)
                    output, err = safe_parse_mapping(content, item["values"])

                if err is not None:
                    # 保守回退：不删除，直接保留 raw 自身
                    output = {v: [v] for v in item["values"]}

                return item["field"], output

        results = await asyncio.gather(*(worker(it) for it in items))

    for field, mapping in results:
        if field not in merged:
            merged[field] = {}
        merged[field].update(mapping)

    with open(out_merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def prepare_inputs(domain: str, field_values_path: str, batch_size: int = 40):
    field_values = load_field_values(field_values_path)
    field_defs = FIELD_DEFS.get(domain, {})

    inputs = []
    for field, values in field_values.items():
        if not values:
            continue

        slot_definition = field_defs.get(field, field)

        batches = get_val_batches(values, batch_size=batch_size)
        for batch in batches:
            # if len(inputs) == 2:
            #     break
            prompt = render_prompt(
                domain=domain,
                slot_name=field,
                slot_definition=slot_definition,
                raw_values=batch
            )
            inputs.append({
                "domain": domain,
                "field": field,
                "values": batch,
                "prompt": prompt
            })

    return inputs


if __name__ == "__main__":
    domains = [
        "electrochemistry",
        "cell_culture",
        "immunotechnology",
        "inorganic_materials",
        "organic_synthesis",
        "polymer_materials"
    ]

    for domain in domains:
        start = time.time()
        
        input_path = f"seed_data/{domain}/open_llm_extraction/field_values.json"
        out_dir = f"seed_data/{domain}/open_llm_cleaning"
        os.makedirs(out_dir, exist_ok=True)

        inputs = prepare_inputs(
            domain=domain,
            field_values_path=input_path,
            batch_size=20
        )
        print(f'domain={domain}, len={len(inputs)}')
        out_path = f"{out_dir}/clean_mapping_merged.json"

        asyncio.run(
            run_all(
                items=inputs,
                out_merged_path=out_path,
                concurrency=32
            )
        )

        print("done:", domain)
        print("merged output ->", out_path)
        print("time:", time.time() - start)

# python -m scripts.structured_extraction.open_llm_clean
        
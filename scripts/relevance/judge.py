import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from openai import OpenAI
from tqdm import tqdm
from json_repair import repair_json

from prompt import *


# =========================
# IO
# =========================
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {path} at line {line_no}: {e}") from e
    return rows



def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Loaders
# =========================
def load_candidate_paragraph_ids(path: str) -> Dict[str, List[str]]:
    obj = read_json(path)
    return {str(candidate_id): [str(pid) for pid in paragraph_ids] for candidate_id, paragraph_ids in obj.items()}



def load_boundary_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row["candidate_id"]): row for row in rows}



def load_paragraphs(path: str) -> Dict[str, str]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Paragraph file must be a JSON list: {path}")

    out: Dict[str, str] = {}
    for row in rows:
        paragraph_id = str(row["id"])
        paragraph_text = str(row["paragraph"]).strip()
        if paragraph_text:
            out[paragraph_id] = paragraph_text
    return out


# =========================
# Existing-result helpers
# =========================
def row_key(candidate_id: str, paragraph_id: str) -> Tuple[str, str]:
    return str(candidate_id), str(paragraph_id)



def load_existing_results(path: str) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    return read_jsonl(path)



def build_existing_key_set(rows: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    for row in rows:
        candidate_id = row.get("candidate_id")
        paragraph_id = row.get("paragraph_id")
        if candidate_id is None or paragraph_id is None:
            continue
        keys.add(row_key(str(candidate_id), str(paragraph_id)))
    return keys


# =========================
# Metadata helpers
# =========================
def get_task_family(type_name: str) -> str:
    return "exploration" if "exploration" in str(type_name) else "reproduction"


# =========================
# Prompt composition
# =========================
def build_user_prompt(
    domain: str,
    task_family: str,
    constraints: List[Dict[str, Any]],
    explore_field: Optional[str],
    paragraph_text: str,
) -> str:
    if task_family == "exploration":
        return build_exploration_prompt(
            domain=domain,
            constraints=constraints,
            explore_field=explore_field,
            paragraph_text=paragraph_text,
        )

    return build_reproduction_prompt(
        domain=domain,
        constraints=constraints,
        paragraph_text=paragraph_text,
    )



def load_api_keys(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        keys = json.load(f)
    if not isinstance(keys, list) or not keys:
        raise ValueError("keys.json must be a non-empty JSON list.")
    return [str(k).strip() for k in keys if str(k).strip()]



def make_clients(api_keys: List[str], base_url: Optional[str] = None) -> List[OpenAI]:
    clients = []
    for key in api_keys:
        kwargs: Dict[str, Any] = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        clients.append(OpenAI(**kwargs))
    return clients



def call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()



def parse_model_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        repaired = repair_json(text)
        return json.loads(repaired)
    except Exception:
        pass

    raise ValueError("Failed to parse model output as JSON")


# =========================
# Row preparation
# =========================
def prepare_judge_rows(
    candidate_paragraph_ids: Dict[str, List[str]],
    boundary_manifest: Dict[str, Dict[str, Any]],
    paragraphs: Dict[str, str],
    existing_keys: Optional[Set[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    existing_keys = existing_keys or set()

    for candidate_id, paragraph_ids in candidate_paragraph_ids.items():
        manifest_row = boundary_manifest.get(candidate_id)
        if manifest_row is None:
            continue

        domain = manifest_row["domain"]
        type_name = manifest_row["type"]
        task_family = get_task_family(type_name)
        constraints = manifest_row.get("constraints", []) or []
        explore_field = manifest_row.get("explore_field")

        for paragraph_id in paragraph_ids:
            key = row_key(candidate_id, paragraph_id)
            if key in existing_keys:
                continue

            paragraph_text = paragraphs.get(str(paragraph_id))
            if not paragraph_text:
                continue

            rows.append(
                {
                    "candidate_paragraph_id": key,
                    "candidate_id": candidate_id,
                    "paragraph_id": str(paragraph_id),
                    "domain": domain,
                    "type": type_name,
                    "type_id": manifest_row.get("type_id"),
                    "task_family": task_family,
                    "candidate_source": manifest_row.get("candidate_source"),
                    "sample_role": "stage2_pool",
                    "error_types": [],
                    "constraints": constraints,
                    "explore_field": explore_field,
                    "paragraph_text": paragraph_text,
                }
            )

    return rows


# =========================
# Worker
# =========================
def process_one(
    idx: int,
    row: Dict[str, Any],
    clients: List[OpenAI],
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep: float,
    save_prompt_text: bool,
) -> Dict[str, Any]:
    client = clients[idx % len(clients)]
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_user_prompt(
        domain=row["domain"],
        task_family=row["task_family"],
        constraints=row["constraints"],
        explore_field=row["explore_field"],
        paragraph_text=row["paragraph_text"],
    )

    parsed = None
    raw_output = None
    last_error = None

    for attempt in range(max_retries):
        try:
            raw_output = call_model(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed = parse_model_json(raw_output)
            last_error = None
            break
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(retry_sleep)

    result = {
        "candidate_id": row["candidate_id"],
        "paragraph_id": row["paragraph_id"],
        "domain": row["domain"],
        "type": row["type"],
        "type_id": row["type_id"],
        "task_family": row["task_family"],
        "candidate_source": row["candidate_source"],
        "sample_role": row["sample_role"],
        "error_types": row["error_types"],
        "constraints": row["constraints"],
        "explore_field": row["explore_field"],
        "model": model,
        "ok": parsed is not None,
        "parsed_output": parsed,
        "raw_output": raw_output,
        "error": last_error,
    }
    if save_prompt_text:
        result["system_prompt"] = system_prompt
        result["user_prompt"] = user_prompt
    return result


# =========================
# Stats
# =========================
def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    ok_count = sum(1 for row in rows if row.get("ok"))

    sample_role_counts = Counter(row.get("sample_role", "unknown") for row in rows)
    task_family_counts = Counter(row.get("task_family", "unknown") for row in rows)
    type_counts = Counter(row.get("type", "unknown") for row in rows)

    paragraph_verdict_counts = Counter()
    setup_verdict_counts = Counter()

    for row in rows:
        parsed = row.get("parsed_output")
        if isinstance(parsed, dict):
            paragraph_verdict = parsed.get("paragraph_verdict")
            if paragraph_verdict is not None:
                paragraph_verdict_counts[str(paragraph_verdict)] += 1

            for setup_judgment in parsed.get("setup_judgments", []) or []:
                if isinstance(setup_judgment, dict):
                    setup_verdict = (
                        setup_judgment.get("setup_verdict")
                        or setup_judgment.get("overall_verdict")
                    )
                    if setup_verdict is not None:
                        setup_verdict_counts[str(setup_verdict)] += 1

    def format_counter_dict(counter_obj: Counter) -> Dict[str, int]:
        return dict(sorted(counter_obj.items()))

    return {
        "total_rows": total,
        "ok_count": ok_count,
        "error_count": total - ok_count,
        "ok_ratio": round(ok_count / total, 6) if total else None,
        "sample_role_counts": format_counter_dict(sample_role_counts),
        "task_family_counts": format_counter_dict(task_family_counts),
        "type_counts": format_counter_dict(type_counts),
        "paragraph_verdict_counts": format_counter_dict(paragraph_verdict_counts),
        "setup_verdict_counts": format_counter_dict(setup_verdict_counts),
    }


# =========================
# Main
# =========================
def run(
    domain: str,
    api_keys_path: str,
    candidate_paragraph_ids_path: str,
    boundary_manifest_path: str,
    paragraph_path: str,
    output_jsonl_path: str,
    output_stats_path: str,
    model: str,
    base_url: Optional[str],
    max_workers: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep: float,
    save_prompt_text: bool,
    mode: str,
) -> None:
    candidate_paragraph_ids = load_candidate_paragraph_ids(candidate_paragraph_ids_path)
    boundary_manifest = load_boundary_manifest(boundary_manifest_path)
    paragraphs = load_paragraphs(paragraph_path)

    existing_results: List[Dict[str, Any]] = []
    existing_keys: Set[Tuple[str, str]] = set()
    if mode == "supplement":
        existing_results = load_existing_results(output_jsonl_path)
        existing_keys = build_existing_key_set(existing_results)

    judge_rows = prepare_judge_rows(
        candidate_paragraph_ids=candidate_paragraph_ids,
        boundary_manifest=boundary_manifest,
        paragraphs=paragraphs,
        existing_keys=existing_keys,
    )

    # judge_rows = judge_rows[:2]

    api_keys = load_api_keys(api_keys_path)
    clients = make_clients(api_keys, base_url)

    new_results: List[Dict[str, Any]] = []
    if judge_rows:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_one,
                    idx,
                    row,
                    clients,
                    model,
                    temperature,
                    max_tokens,
                    max_retries,
                    retry_sleep,
                    save_prompt_text,
                )
                for idx, row in enumerate(judge_rows)
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Judging {domain}"):
                new_results.append(future.result())

    merged_results = existing_results + new_results if mode == "supplement" else new_results
    merged_results.sort(key=lambda row: (row["candidate_id"], row["paragraph_id"]))
    stats = summarize_results(merged_results)
    stats["mode"] = mode
    stats["existing_result_count"] = len(existing_results)
    stats["new_result_count"] = len(new_results)
    stats["final_result_count"] = len(merged_results)
    stats["skipped_existing_count"] = len(existing_keys)

    write_jsonl(output_jsonl_path, merged_results)
    write_json(output_stats_path, stats)

    print(f"[OK] Domain: {domain}")
    print(f"[OK] Mode: {mode}")
    print(f"[OK] Existing rows loaded: {len(existing_results)}")
    print(f"[OK] Rows to judge this run: {len(judge_rows)}")
    print(f"[OK] New rows written this run: {len(new_results)}")
    print(f"[OK] Final rows in output: {len(merged_results)}")
    print(f"[OK] Saved results -> {output_jsonl_path}")
    print(f"[OK] Saved stats   -> {output_stats_path}")



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--api_keys_path", default="keys.txt")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--save_prompt_text", action="store_true")
    parser.add_argument("--mode", choices=["overwrite", "supplement"], default="overwrite")

    parser.add_argument("--candidate_paragraph_ids_path", default=None)
    parser.add_argument("--boundary_manifest_path", default=None)
    parser.add_argument("--paragraph_path", default=None)
    parser.add_argument("--output_jsonl_path", default=None)
    parser.add_argument("--output_stats_path", default=None)
    return parser



def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run(
        domain=args.domain,
        api_keys_path=args.api_keys_path,
        candidate_paragraph_ids_path=args.candidate_paragraph_ids_path,
        boundary_manifest_path=args.boundary_manifest_path,
        paragraph_path=args.paragraph_path,
        output_jsonl_path=args.output_jsonl_path,
        output_stats_path=args.output_stats_path,
        model=args.model,
        base_url=args.base_url,
        max_workers=args.max_workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        save_prompt_text=args.save_prompt_text,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

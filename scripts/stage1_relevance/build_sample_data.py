import json
from pathlib import Path
from collections import defaultdict

INPUT_AUDIT_FILE = Path("data/audit_sample_from_paragraph_labels.jsonl")
OUTPUT_FILE = Path("data/audit_sample_data.jsonl")

DATA_ROOT = Path("data")
SEED_ROOT = Path("seed_data")

JUDGE_FILENAME = "judge_results_stage1_gpt-5.4.jsonl"

KEEP_FULL_JUDGE_RECORD = False


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: {path}, line {line_no}") from e
    return records


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_domain_cache(domain: str):
    """
    为单个 domain 构建 3 个索引：
    1. candidate_id -> query_record
    2. paragraph_id -> paragraph_record
    3. (candidate_id, paragraph_id) -> judge_record
    """
    query_path = DATA_ROOT / domain / "gen_query" / "queries_raw.json"
    paragraph_path = SEED_ROOT / domain / "paragraphs.json"
    judge_path = DATA_ROOT / domain / "stage1_relevance" / JUDGE_FILENAME

    # 1) queries
    query_records = load_json(query_path)
    query_index = {str(k): v for k, v in query_records.items()}

    # 2) paragraphs
    paragraph_records = load_json(paragraph_path)
    paragraph_index = {}
    for rec in paragraph_records:
        pid = str(rec.get("id"))
        paragraph_index[pid] = rec

    # 3) judge results
    judge_records = load_jsonl(judge_path)
    judge_index = {}
    duplicate_keys = 0

    for rec in judge_records:
        key = (str(rec.get("candidate_id")), str(rec.get("paragraph_id")))
        if key in judge_index:
            duplicate_keys += 1
        judge_index[key] = rec

    if duplicate_keys:
        print(f"[WARN] domain={domain}: judge 文件中发现 {duplicate_keys} 个重复 (candidate_id, paragraph_id)，已以后者覆盖前者。")

    return {
        "query_index": query_index,
        "paragraph_index": paragraph_index,
        "judge_index": judge_index,
    }


def build_llm_judge_summary(judge_rec):
    if judge_rec is None:
        return None

    parsed_output = judge_rec.get("parsed_output", {}) or {}

    return {
        "model": judge_rec.get("model"),
        "ok": judge_rec.get("ok"),
        "candidate_source": judge_rec.get("candidate_source"),
        "sample_role": judge_rec.get("sample_role"),
        "error_types": judge_rec.get("error_types", []),
        "constraints": judge_rec.get("constraints"),
        "explore_field": judge_rec.get("explore_field"),
        "paragraph_verdict": parsed_output.get("paragraph_verdict"),
        "paragraph_reason": parsed_output.get("paragraph_reason"),
        "setup_judgments": parsed_output.get("setup_judgments"),
    }


def enrich_record(rec, domain_cache):
    candidate_id = str(rec["candidate_id"])
    paragraph_id = str(rec["paragraph_id"])

    query_rec = domain_cache["query_index"].get(candidate_id)
    paragraph_rec = domain_cache["paragraph_index"].get(paragraph_id)
    judge_rec = domain_cache["judge_index"].get((candidate_id, paragraph_id))

    enriched = dict(rec)

    # ---- query 信息 ----
    if query_rec is not None:
        enriched["query"] = query_rec.get("query")
        enriched["query_meta"] = {
            "domain": query_rec.get("domain"),
            "type": query_rec.get("type"),
            "type_id": query_rec.get("type_id"),
            "constraint_count": query_rec.get("constraint_count"),
            "has_numeric": query_rec.get("has_numeric"),
            "has_negation": query_rec.get("has_negation"),
            "has_range": query_rec.get("has_range"),
        }
    else:
        enriched["query"] = None
        enriched["query_meta"] = None

    # ---- paragraph 信息 ----
    if paragraph_rec is not None:
        enriched["paragraph_meta"] = {
            "id": paragraph_rec.get("id"),
            "doi": paragraph_rec.get("doi"),
            "head": paragraph_rec.get("head"),
            "source": paragraph_rec.get("source"),
        }
        enriched["paragraph_text"] = paragraph_rec.get("paragraph")
    else:
        enriched["paragraph_meta"] = None
        enriched["paragraph_text"] = None

    # ---- LLM judge 信息 ----
    enriched["llm_judge_summary"] = build_llm_judge_summary(judge_rec)

    if KEEP_FULL_JUDGE_RECORD:
        enriched["llm_judge_record"] = judge_rec

    # ---- 缺失标记 ----
    missing = {
        "missing_query": query_rec is None,
        "missing_paragraph": paragraph_rec is None,
        "missing_judge": judge_rec is None,
    }

    return enriched, missing


def main():
    audit_records = load_jsonl(INPUT_AUDIT_FILE)
    print(f"读取审计样本: {len(audit_records)} 条")

    domains = sorted({rec["domain"] for rec in audit_records})
    print("涉及 domain:", domains)

    domain_caches = {}
    for domain in domains:
        domain_caches[domain] = build_domain_cache(domain)

    stats = defaultdict(int)
    domain_stats = defaultdict(lambda: defaultdict(int))
    enriched_records = []

    for rec in audit_records:
        domain = rec["domain"]
        enriched, miss = enrich_record(rec, domain_caches[domain])
        enriched_records.append(enriched)

        stats["total"] += 1
        domain_stats[domain]["total"] += 1

        for k, v in miss.items():
            if v:
                stats[k] += 1
                domain_stats[domain][k] += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for rec in enriched_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"已写出: {OUTPUT_FILE}")
    print("\n总体统计:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\n分 domain 统计:")
    print(json.dumps(domain_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# python scripts/stage1_relevance/build_sample_data.py


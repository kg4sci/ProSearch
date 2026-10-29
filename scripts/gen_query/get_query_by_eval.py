import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import json_repair
import random

def load_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        print(f"[skip] file not found: {path}")
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")


def safe_rate(num, den):
    return round(num / den, 4) if den else 0.0


def parse_embedded_json(text):
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json_repair.loads(text)
    except Exception:
        return None


def build_generated_lookup(generated_rows):
    """
    Build candidate_id -> metadata lookup, especially skip_reason from gen_raw_response.
    """
    lookup = {}

    for row in generated_rows:
        candidate_id = row.get("candidate_id")
        if not candidate_id:
            continue

        parsed = parse_embedded_json(row.get("gen_raw_response"))

        skip_reason = row.get("skip_reason")
        if skip_reason is None and parsed:
            skip_reason = parsed.get("skip_reason")

        request_query = row.get("request_query")
        if request_query is None and parsed:
            request_query = parsed.get("request_query")

        skip = row.get("skip")
        if skip is None and parsed:
            skip = parsed.get("skip")

        lookup[candidate_id] = {
            "skip_reason": skip_reason,
            "request_query": request_query,
            "skip": skip,
            "row": row,
        }

    return lookup


def build_sampled_lookup(sampled_rows):
    """
    Build candidate_id -> sampled constraint metadata lookup.
    """
    lookup = {}

    for row in sampled_rows:
        candidate_id = row.get("candidate_id")
        if not candidate_id:
            continue

        lookup[candidate_id] = {
            "constraint_count": row.get("constraint_count", row.get("total")),
            "has_numeric": row.get("has_numeric_constraint", row.get("has_numeric")),
            "has_negation": row.get("has_negation"),
            "has_range": row.get("has_range_constraint", False),
            "type": row.get("type"),
            "type_id": row.get("type_id"),
            "domain": row.get("domain"),
            "explore_field": row.get("explore_field"),
            "constraints": row.get("constraints"),
            "row": row,
        }

    return lookup


def enrich_evaluated_rows(eval_rows, generated_lookup):
    """
    Add skip_reason from generated file when it is absent in evaluated file.
    """
    enriched = []

    for row in eval_rows:
        row = dict(row)
        candidate_id = row.get("candidate_id")
        gen_info = generated_lookup.get(candidate_id, {})

        if row.get("skip_reason") is None:
            row["skip_reason"] = gen_info.get("skip_reason")

        if row.get("request_query") is None and gen_info.get("request_query") is not None:
            row["request_query"] = gen_info.get("request_query")

        if row.get("skip") is None and gen_info.get("skip") is not None:
            row["skip"] = gen_info.get("skip")

        enriched.append(row)

    return enriched


def is_generated_query(row):
    return (row.get("skip") is False) and bool(row.get("request_query"))


def is_skipped_query(row):
    return (row.get("skip") is True) or (row.get("eval_status") == "skipped")


def summarize_distribution(rows, key, default="unknown"):
    counter = Counter(r.get(key, default) for r in rows)
    total = len(rows)
    return {
        k: {
            "count": v,
            "rate": safe_rate(v, total)
        }
        for k, v in sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))
    }


def summarize_generated_queries(rows):
    """
    Summarize only rows with generated queries.
    """
    total = len(rows)

    faithfulness_1 = sum(1 for r in rows if r.get("faithfulness") == 1)
    clarity_1 = sum(1 for r in rows if r.get("clarity") == 1)
    retrieval_1 = sum(1 for r in rows if r.get("retrieval_usefulness") == 1)

    all_1 = sum(
        1 for r in rows
        if r.get("faithfulness") == 1
        and r.get("clarity") == 1
        and r.get("retrieval_usefulness") == 1
    )

    keep_n = sum(1 for r in rows if r.get("final_decision") == "keep")
    revise_n = sum(1 for r in rows if r.get("final_decision") == "revise")
    drop_n = sum(1 for r in rows if r.get("final_decision") == "drop")

    return {
        "n_generated_queries": total,
        "faithfulness_rate": safe_rate(faithfulness_1, total),
        "clarity_rate": safe_rate(clarity_1, total),
        "retrieval_usefulness_rate": safe_rate(retrieval_1, total),
        "all_three_pass_count": all_1,
        "all_three_pass_rate": safe_rate(all_1, total),
        "keep_count": keep_n,
        "keep_rate": safe_rate(keep_n, total),
        "revise_count": revise_n,
        "revise_rate": safe_rate(revise_n, total),
        "drop_count": drop_n,
        "drop_rate": safe_rate(drop_n, total),
        "primary_issue_distribution": summarize_distribution(rows, "primary_issue"),
        "final_decision_distribution": summarize_distribution(rows, "final_decision"),
    }


def summarize_overall(rows):
    total = len(rows)
    skipped_rows = [r for r in rows if is_skipped_query(r)]
    generated_rows = [r for r in rows if is_generated_query(r)]

    return {
        "n_total_records": total,
        "n_skipped": len(skipped_rows),
        "skip_rate": safe_rate(len(skipped_rows), total),
        "skip_reason_distribution": summarize_distribution(skipped_rows, "skip_reason"),
        "n_generated_queries": len(generated_rows),
        "generated_rate": safe_rate(len(generated_rows), total),
        "generated_query_summary": summarize_generated_queries(generated_rows),
    }


def summarize_by_type(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("type", "unknown")].append(r)

    result = {}
    for t, items in sorted(groups.items()):
        result[t] = summarize_overall(items)
    return result


def collect_bad_cases(rows, limit=30):
    generated_rows = [r for r in rows if is_generated_query(r)]

    bad = []
    for r in generated_rows:
        faithfulness = r.get("faithfulness")
        clarity = r.get("clarity")
        retrieval = r.get("retrieval_usefulness")

        if not (faithfulness == 1 and clarity == 1 and retrieval == 1):
            bad.append({
                "candidate_id": r.get("candidate_id"),
                "domain": r.get("domain"),
                "type": r.get("type"),
                "type_id": r.get("type_id"),
                "constraints": r.get("constraints"),
                "explore_field": r.get("explore_field"),
                "request_query": r.get("request_query"),
                "faithfulness": faithfulness,
                "clarity": clarity,
                "retrieval_usefulness": retrieval,
                "primary_issue": r.get("primary_issue"),
                "final_decision": r.get("final_decision"),
                "reason": r.get("reason"),
                "suggested_revision": r.get("suggested_revision", "")
            })

    # bad = sorted(
    #     bad,
    #     key=lambda x: (
    #         x.get("primary_issue", ""),
    #         x.get("type", ""),
    #         x.get("candidate_id", "")
    #     )
    # )
    return bad[:limit]


def build_queries_raw(eval_rows, sampled_lookup):
    result = {}

    for row in eval_rows:
        final_decision = row.get("final_decision")
        if final_decision not in {"keep", "revise"}:
            continue

        candidate_id = row.get("candidate_id")
        if not candidate_id:
            continue

        if final_decision == "keep":
            query = row.get("request_query")
        else:
            query = row.get("suggested_revision") or row.get("request_query")

        if not query:
            continue

        sampled_info = sampled_lookup.get(candidate_id, {})

        result[candidate_id] = {
            "query": query,
            "domain": row.get("domain", sampled_info.get("domain")),
            "type": row.get("type", sampled_info.get("type")),
            "type_id": row.get("type_id", sampled_info.get("type_id")),
            "constraint_count": sampled_info.get("constraint_count"),
            "has_numeric": sampled_info.get("has_numeric"),
            "has_negation": sampled_info.get("has_negation"),
            "has_range": sampled_info.get("has_range"),
        }

    return result


def stratified_sample_queries(queries_raw, total_n=20, seed=42):
    random.seed(seed)

    items = []
    for cid, data in queries_raw.items():
        row = dict(data)
        row["candidate_id"] = cid
        items.append(row)

    type_groups = defaultdict(list)
    for r in items:
        t = r.get("type", "unknown")
        type_groups[t].append(r)

    types = list(type_groups.keys())
    n_types = len(types)

    base = total_n // n_types
    remainder = total_n % n_types

    type_quota = {t: base for t in types}
    for t in types[:remainder]:
        type_quota[t] += 1

    sampled = []

    for t in types:
        group = type_groups[t]
        quota = type_quota[t]

        cc_groups = defaultdict(list)
        for r in group:
            cc = r.get("constraint_count")
            cc_groups[cc].append(r)

        cc_values = list(cc_groups.keys())
        cc_base = quota // len(cc_values) if cc_values else 0
        cc_rem = quota % len(cc_values) if cc_values else 0

        cc_quota = {cc: cc_base for cc in cc_values}
        for cc in cc_values[:cc_rem]:
            cc_quota[cc] += 1

        selected = []

        # 先按 cc 分配采样
        for cc in cc_values:
            candidates = cc_groups[cc]
            k = min(len(candidates), cc_quota[cc])
            selected.extend(random.sample(candidates, k))

        # 如果不够 quota，用剩余补
        if len(selected) < quota:
            remaining = [r for r in group if r not in selected]
            need = quota - len(selected)
            if remaining:
                selected.extend(random.sample(remaining, min(len(remaining), need)))

        sampled.extend(selected)

    sampled = sampled[:total_n]

    result = {
        r["candidate_id"]: {
            k: v for k, v in r.items() if k != "candidate_id"
        }
        for r in sampled
    }

    return result

def build_stats(eval_rows, bad_case_limit=30):
    return {
        "overall": summarize_overall(eval_rows),
        "by_type": summarize_by_type(eval_rows),
        "bad_cases": collect_bad_cases(eval_rows, limit=bad_case_limit),
    }


if __name__ == "__main__":
    domains = [
        # "electrochemistry",
        "cell_culture",
        "immunotechnology",
        "inorganic_materials",
        "organic_synthesis",
        "polymer_materials"
    ]

    bad_case_limit = 30

    for d in domains:
        print(f"Processing domain: {d}")

        eval_path = f"data/{d}/gen_query/evaluated_queries.jsonl"
        gen_path = f"data/{d}/gen_query/generated_queries.jsonl"
        sampled_path = f"seed_data/{d}/constraints/sampled_constraints_stage1.jsonl"

        stats_output_path = f"data/{d}/gen_query/query_eval_stats.json"
        raw_output_path = f"data/{d}/gen_query/queries_raw.json"

        eval_rows = load_jsonl(eval_path)
        gen_rows = load_jsonl(gen_path)
        sampled_rows = load_jsonl(sampled_path)

        if not eval_rows:
            print(f"[skip] no evaluated rows for {d}")
            continue

        gen_lookup = build_generated_lookup(gen_rows)
        sampled_lookup = build_sampled_lookup(sampled_rows)

        eval_rows = enrich_evaluated_rows(eval_rows, gen_lookup)

        stats = build_stats(eval_rows, bad_case_limit=bad_case_limit)
        queries_raw_full = build_queries_raw(eval_rows, sampled_lookup)

        print(f"Stratified sampling {len(queries_raw_full)} queries...")
        queries_raw = stratified_sample_queries(
            queries_raw_full,
            total_n=15,  
            seed=42
        )
        save_json(stats, stats_output_path)
        # save_json(queries_raw, raw_output_path)


# python scripts/gen_query/get_query_by_eval.py


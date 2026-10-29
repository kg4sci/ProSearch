import json
import argparse
from collections import OrderedDict


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse JSONL line {line_no} in {path}: {e}")


def build_generated_query_index(generated_queries_path):
    """
    generated_queries.jsonl -> {candidate_id: generated_record}
    """
    index = {}

    for rec in load_jsonl(generated_queries_path):
        candidate_id = rec.get("candidate_id")
        if not candidate_id:
            continue
        index[candidate_id] = rec

    return index


def build_queries_raw(boundary_manifest_path, generated_queries_path):
    gen_index = build_generated_query_index(generated_queries_path)

    queries_raw = OrderedDict()

    total = 0
    matched = 0
    missing_gen = 0
    skipped = 0

    for rec in load_jsonl(boundary_manifest_path):
        total += 1

        candidate_id = rec["candidate_id"]
        gen_rec = gen_index.get(candidate_id)

        query = ""
        if gen_rec is None:
            missing_gen += 1
        else:
            if gen_rec.get("skip", False):
                skipped += 1
                query = ""
            else:
                matched += 1
                query = gen_rec.get("request_query") or ""

        out = {
            "query": query,
            "domain": rec.get("domain"),
            "type": rec.get("type"),
            "type_id": rec.get("type_id"),
            "constraint_count": len(rec.get("constraints", []) or []),
            "has_numeric": rec.get("has_numeric_constraint", False),
            "has_negation": rec.get("has_negation", False),
            "has_range": rec.get("has_range_constraint", False),
        }

        # exploration 任务保留 explore_field；reproduction 没有也没关系
        if "explore_field" in rec:
            out["explore_field"] = rec.get("explore_field")

        # 这些字段通常也很有用，保留着后面做分析方便
        if "candidate_source" in rec:
            out["candidate_source"] = rec.get("candidate_source")
        if "support_count" in rec:
            out["support_count"] = rec.get("support_count")

        queries_raw[candidate_id] = out

    stats = {
        "total_candidates": total,
        "matched_generated_queries": matched,
        "missing_generated_queries": missing_gen,
        "skipped_generated_queries": skipped,
    }

    return queries_raw, stats


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
        generated_queries_path = f"data/{domain}/gen_query/generated_queries.jsonl"
        boundary_manifest_path = f"seed_data/{domain}/constraints/sampled_constraints_stage1.jsonl"
        output_path = f"data/{domain}/gen_query/queries_raw.json"
        
        queries_raw, stats = build_queries_raw(
            boundary_manifest_path=boundary_manifest_path,
            generated_queries_path=generated_queries_path,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(queries_raw, f, ensure_ascii=False, indent=2)

        print(f"[Done] Saved queries_raw.json to: {output_path}")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

# python scripts/gen_query/get_query_by_gen.py


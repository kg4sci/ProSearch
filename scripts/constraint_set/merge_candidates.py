
import json
from pathlib import Path
from collections import Counter


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_key(candidate):
    payload = {
        "domain": candidate.get("domain"),
        "type": candidate.get("type"),
        "type_id": candidate.get("type_id"),
        "has_negation": bool(candidate.get("has_negation", False)),
        "explore_field": candidate.get("explore_field"),
        "constraints": sorted(
            candidate.get("constraints", []),
            key=lambda x: (x.get("field"), x.get("op"), str(x.get("value", "")), str(x.get("range", "")))
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_field_registry(path):
    data = read_json(path)
    return data.get("fields", data)


def enrich_candidate(candidate, field_registry):
    constraints = candidate.get("constraints", [])

    numeric_fields = []
    range_fields = []
    negated_fields = []

    for lit in constraints:
        field = lit.get("field")
        cfg = field_registry.get(field, {})
        if cfg.get("value_type") == "numeric":
            numeric_fields.append(field)
        if lit.get("op") == "between":
            range_fields.append(field)
        if lit.get("op") == "!=":
            negated_fields.append(field)

    candidate["numeric_constraint_fields"] = sorted(set(numeric_fields))
    candidate["numeric_constraint_count"] = len(candidate["numeric_constraint_fields"])
    candidate["has_numeric_constraint"] = candidate["numeric_constraint_count"] > 0

    candidate["range_constraint_fields"] = sorted(set(range_fields))
    candidate["range_constraint_count"] = len(candidate["range_constraint_fields"])
    candidate["has_range_constraint"] = candidate["range_constraint_count"] > 0

    candidate["negated_constraint_fields"] = sorted(set(negated_fields))
    candidate["negated_constraint_count"] = len(candidate["negated_constraint_fields"])

    return candidate


def merge_candidates(
    positive_path,
    negative_path,
    field_registry_path,
    merged_output_path,
    stats_output_path,
):
    field_registry = load_field_registry(field_registry_path)

    merged = []
    seen = set()

    source_counts = Counter()
    type_counts = Counter()
    type_id_counts = Counter()
    negation_counts = Counter()
    support_counts = Counter()
    explore_bucket_counts = Counter()

    numeric_candidate_counts = Counter()
    range_candidate_counts = Counter()
    numeric_constraint_slot_counts = Counter()
    range_constraint_slot_counts = Counter()

    duplicate_positive = 0
    duplicate_negative = 0

    for source_name, path in [("positive", positive_path), ("negative", negative_path)]:
        if not Path(path).exists():
            print(f"[WARN] File not found, skip: {path}")
            continue

        for row in read_jsonl(path):
            row = dict(row)
            row["candidate_source"] = source_name
            row["has_negation"] = bool(row.get("has_negation", False))

            row = enrich_candidate(row, field_registry)

            k = candidate_key(row)
            if k in seen:
                if source_name == "positive":
                    duplicate_positive += 1
                else:
                    duplicate_negative += 1
                continue

            seen.add(k)
            merged.append(row)

            source_counts[source_name] += 1
            type_counts[row.get("type")] += 1
            type_id_counts[row.get("type_id")] += 1
            negation_counts[str(row.get("has_negation", False))] += 1
            support_counts[str(row.get("support"))] += 1

            explore_bucket = row.get("explore_field")
            if explore_bucket is None:
                explore_bucket = "none"
            explore_bucket_counts[explore_bucket] += 1

            numeric_candidate_counts[str(row.get("has_numeric_constraint", False))] += 1
            range_candidate_counts[str(row.get("has_range_constraint", False))] += 1
            numeric_constraint_slot_counts[str(row.get("numeric_constraint_count", 0))] += 1
            range_constraint_slot_counts[str(row.get("range_constraint_count", 0))] += 1

    for i, row in enumerate(merged, start=1):
        domain = row.get("domain", "unknown")
        prefix = "neg" if row.get("has_negation", False) else "pos"
        row["stage2_candidate_id"] = f"{domain}_stage2_{prefix}_{i:06d}"

    stats = {
        "total_candidates": len(merged),
        "source_counts": dict(source_counts),
        "type_counts": dict(type_counts),
        "type_id_counts": dict(type_id_counts),
        "negation_counts": dict(negation_counts),
        "explore_bucket_counts": dict(explore_bucket_counts),
        "support_counts": dict(sorted(support_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])),
        "numeric_candidate_counts": dict(numeric_candidate_counts),
        "range_candidate_counts": dict(range_candidate_counts),
        "numeric_constraint_slot_counts": dict(sorted(numeric_constraint_slot_counts.items(), key=lambda x: int(x[0]))),
        "range_constraint_slot_counts": dict(sorted(range_constraint_slot_counts.items(), key=lambda x: int(x[0]))),
        "duplicate_skipped": {
            "positive": duplicate_positive,
            "negative": duplicate_negative,
            "total": duplicate_positive + duplicate_negative,
        },
    }

    write_jsonl(merged_output_path, merged)
    write_json(stats_output_path, stats)

    print(f"[OK] Saved merged candidates: {merged_output_path}")
    print(f"[OK] Saved merge stats: {stats_output_path}")
    print(f"[OK] Total merged candidates: {len(merged)}")


if __name__ == "__main__":
    domains = [
        # "electrochemistry",
        # "cell_culture",
        "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        # "polymer_materials"
    ]

    for d in domains:
        print(f"\n=== Merging: {d} ===")
        positive_path = f"seed_data/{d}/constraints/constraint_candidates_positive.jsonl"
        negative_path = f"seed_data/{d}/constraints/constraint_candidates_negative.jsonl"
        field_registry_path = f"seed_data/{d}/field_registry.json"

        merged_output_path = f"seed_data/{d}/constraints/constraint_candidates_merged.jsonl"
        stats_output_path = f"seed_data/{d}/constraints/constraint_candidates_merged.stats.json"

        merge_candidates(
            positive_path=positive_path,
            negative_path=negative_path,
            field_registry_path=field_registry_path,
            merged_output_path=merged_output_path,
            stats_output_path=stats_output_path,
        )


# python scripts/constraint_set/merge_candidates.py
        
        
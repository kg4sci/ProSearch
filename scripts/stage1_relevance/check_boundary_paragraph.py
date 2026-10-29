import json
from collections import defaultdict
from pathlib import Path


BOUNDARY_TYPES = [
    "fixed_constraint_missing_1",
    "fixed_constraint_mismatch_1",
    "negated_field_mismatch",
    "negated_field_missing",
    "explore_field_missing",
]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Helpers
# =========================
def sort_ids(ids):
    ids = list(set(str(x) for x in ids))
    def key_fn(x):
        return (0, int(x)) if x.isdigit() else (1, x)
    return sorted(ids, key=key_fn)


def safe_json_loads(x):
    if isinstance(x, dict):
        return x
    if not x:
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def parse_judge_output(record):
    parsed = safe_json_loads(record.get("parsed_output"))
    if parsed is not None:
        return parsed

    # fallback
    parsed = safe_json_loads(record.get("raw_output"))
    return parsed


def get_task_family(record):
    tf = record.get("task_family")
    if tf:
        return tf
    return "exploration" if record.get("explore_field") else "reproduction"


def normalize_fixed_status(status):
    s = str(status or "").strip().lower()
    if s in {"match", "matched"}:
        return "match"
    if s in {"missing", "absent", "unknown", "not_found"}:
        return "missing"
    if s in {"mismatch", "violate", "violated", "conflict", "incompatible"}:
        return "mismatch"
    return "missing"


def normalize_explore_status(status):
    s = str(status or "").strip().lower()
    if s in {"present", "provided", "found"}:
        return "present"
    return "missing"


def split_constraints(constraints):
    constraints = constraints or []
    negated_constraints = [c for c in constraints if c.get("op") == "!="]
    non_negated_constraints = [c for c in constraints if c.get("op") != "!="]
    return constraints, negated_constraints, non_negated_constraints


# =========================
# Setup-level classification
# =========================
def classify_setup_from_judge(setup_judgment, constraints, task_family):
    """
    setup_label ∈ {
        exact_support,
        fixed_constraint_missing_1,
        fixed_constraint_mismatch_1,
        negated_field_mismatch,
        negated_field_missing,
        explore_field_missing,
        non_boundary
    }
    """
    constraints, negated_constraints, non_negated_constraints = split_constraints(constraints)

    fixed_field_judgments = setup_judgment.get("fixed_field_judgments", []) or []
    status_map = {}

    for item in fixed_field_judgments:
        field = item.get("field")
        if not field:
            continue
        status_map[field] = normalize_fixed_status(item.get("status"))

    # judge 如果漏掉某个约束字段，按 missing 处理
    for lit in constraints:
        status_map.setdefault(lit["field"], "missing")

    explore_status = None
    if task_family == "exploration":
        explore_j = setup_judgment.get("explore_field_judgment", {}) or {}
        explore_status = normalize_explore_status(explore_j.get("status"))

    all_constraints_match = all(status_map[lit["field"]] == "match" for lit in constraints)

    # 1) exact support
    if task_family == "reproduction":
        if all_constraints_match:
            return {
                "setup_label": "exact_support",
                "status_map": status_map,
                "explore_status": None,
            }
    else:
        if all_constraints_match and explore_status == "present":
            return {
                "setup_label": "exact_support",
                "status_map": status_map,
                "explore_status": explore_status,
            }

    if negated_constraints:
        # 2.1 negated_field_mismatch
        for lit in negated_constraints:
            f = lit["field"]
            other_ok = all(
                status_map[other["field"]] == "match"
                for other in constraints
                if other["field"] != f
            )
            if other_ok and status_map[f] == "mismatch":
                return {
                    "setup_label": "negated_field_mismatch",
                    "status_map": status_map,
                    "explore_status": explore_status,
                }

        # 2.2 negated_field_missing
        for lit in negated_constraints:
            f = lit["field"]
            other_ok = all(
                status_map[other["field"]] == "match"
                for other in constraints
                if other["field"] != f
            )
            if other_ok and status_map[f] == "missing":
                return {
                    "setup_label": "negated_field_missing",
                    "status_map": status_map,
                    "explore_status": explore_status,
                }

    # 3) fixed near-miss
    if negated_constraints:
        neg_all_match = all(status_map[lit["field"]] == "match" for lit in negated_constraints)
        if neg_all_match:
            missing_fields = [
                lit["field"]
                for lit in non_negated_constraints
                if status_map[lit["field"]] == "missing"
            ]
            mismatch_fields = [
                lit["field"]
                for lit in non_negated_constraints
                if status_map[lit["field"]] == "mismatch"
            ]

            if len(missing_fields) + len(mismatch_fields) == 1:
                return {
                    "setup_label": (
                        "fixed_constraint_missing_1"
                        if len(missing_fields) == 1
                        else "fixed_constraint_mismatch_1"
                    ),
                    "status_map": status_map,
                    "explore_status": explore_status,
                }
    else:
        missing_fields = [
            lit["field"]
            for lit in constraints
            if status_map[lit["field"]] == "missing"
        ]
        mismatch_fields = [
            lit["field"]
            for lit in constraints
            if status_map[lit["field"]] == "mismatch"
        ]

        if len(missing_fields) + len(mismatch_fields) == 1:
            return {
                "setup_label": (
                    "fixed_constraint_missing_1"
                    if len(missing_fields) == 1
                    else "fixed_constraint_mismatch_1"
                ),
                "status_map": status_map,
                "explore_status": explore_status,
            }

    # 4) explore_field_missing
    # 与旧代码口径一致：所有 constraints 都 match，但 explore field 缺失
    if task_family == "exploration" and all_constraints_match and explore_status == "missing":
        return {
            "setup_label": "explore_field_missing",
            "status_map": status_map,
            "explore_status": explore_status,
        }

    # 5) 非边界 setup
    return {
        "setup_label": "non_boundary",
        "status_map": status_map,
        "explore_status": explore_status,
    }


# =========================
# Paragraph-level classification
# =========================
def classify_paragraph_from_judge(record):
    """
    paragraph_label ∈ {
        exact_support,
        boundary_negative,
        non_boundary_negative,
        unjudged
    }
    """
    parsed = parse_judge_output(record)
    if parsed is None:
        return {
            "paragraph_label": "unjudged",
            "boundary_types": [],
            "setup_results": [],
            "judge_paragraph_verdict": None,
        }

    constraints = record.get("constraints", []) or []
    task_family = get_task_family(record)
    setup_judgments = parsed.get("setup_judgments", []) or []

    setup_results = [
        classify_setup_from_judge(sj, constraints, task_family)
        for sj in setup_judgments
    ]

    # paragraph-level 正样本优先
    if any(r["setup_label"] == "exact_support" for r in setup_results):
        return {
            "paragraph_label": "exact_support",
            "boundary_types": [],
            "setup_results": setup_results,
            "judge_paragraph_verdict": parsed.get("paragraph_verdict"),
        }

    boundary_types = sorted({
        r["setup_label"]
        for r in setup_results
        if r["setup_label"] in BOUNDARY_TYPES
    })

    if boundary_types:
        return {
            "paragraph_label": "boundary_negative",
            "boundary_types": boundary_types,
            "setup_results": setup_results,
            "judge_paragraph_verdict": parsed.get("paragraph_verdict"),
        }

    return {
        "paragraph_label": "non_boundary_negative",
        "boundary_types": [],
        "setup_results": setup_results,
        "judge_paragraph_verdict": parsed.get("paragraph_verdict"),
    }


# =========================
# Main aggregation
# =========================
def relabel_from_judge(
    input_path,
    paragraph_output_path,
    candidate_output_path,
):
    paragraph_rows = []
    candidate_map = {}

    for record in read_jsonl(input_path):
        cid = record["candidate_id"]
        pid = str(record["paragraph_id"])

        cls = classify_paragraph_from_judge(record)

        # paragraph-level row
        paragraph_row = {
            "candidate_id": cid,
            "paragraph_id": pid,
            "domain": record.get("domain"),
            "type": record.get("type"),
            "type_id": record.get("type_id"),
            "task_family": get_task_family(record),
            "paragraph_label": cls["paragraph_label"],
            "boundary_types": cls["boundary_types"],
            "judge_paragraph_verdict": cls["judge_paragraph_verdict"],
            "setup_labels": [x["setup_label"] for x in cls["setup_results"]],
        }
        paragraph_rows.append(paragraph_row)

        # candidate-level init
        if cid not in candidate_map:
            candidate_map[cid] = {
                "candidate_id": cid,
                "domain": record.get("domain"),
                "type": record.get("type"),
                "type_id": record.get("type_id"),
                "task_family": get_task_family(record),
                "constraints": record.get("constraints"),
                "explore_field": record.get("explore_field"),
                "has_negation": record.get("has_negation"),
                "has_numeric_constraint": record.get("has_numeric_constraint"),
                "has_range_constraint": record.get("has_range_constraint"),
                "candidate_source": record.get("candidate_source"),
                "support_paragraph_ids": set(),
                "judged_boundary_paragraph_ids": {k: set() for k in BOUNDARY_TYPES},
                "non_boundary_negative_paragraph_ids": set(),
                "unjudged_paragraph_ids": set(),
            }

        agg = candidate_map[cid]

        if cls["paragraph_label"] == "exact_support":
            agg["support_paragraph_ids"].add(pid)

        elif cls["paragraph_label"] == "boundary_negative":
            for b in cls["boundary_types"]:
                agg["judged_boundary_paragraph_ids"][b].add(pid)

        elif cls["paragraph_label"] == "non_boundary_negative":
            agg["non_boundary_negative_paragraph_ids"].add(pid)

        else:
            agg["unjudged_paragraph_ids"].add(pid)

    # write paragraph-level output
    paragraph_rows.sort(key=lambda x: (str(x["candidate_id"]), str(x["paragraph_id"])))
    write_jsonl(paragraph_output_path, paragraph_rows)

    # write candidate-level output
    candidate_rows = []
    for cid in sorted(candidate_map.keys()):
        agg = candidate_map[cid]
        row = {
            "candidate_id": agg["candidate_id"],
            "domain": agg["domain"],
            "type": agg["type"],
            "type_id": agg["type_id"],
            "task_family": agg["task_family"],
            "constraints": agg["constraints"],
            "explore_field": agg["explore_field"],
            "has_negation": agg["has_negation"],
            "has_numeric_constraint": agg["has_numeric_constraint"],
            "has_range_constraint": agg["has_range_constraint"],
            "candidate_source": agg["candidate_source"],
            "support_paragraph_ids": sort_ids(agg["support_paragraph_ids"]),
            "support_count": len(agg["support_paragraph_ids"]),
            "judged_boundary_paragraph_ids": {
                k: sort_ids(v) for k, v in agg["judged_boundary_paragraph_ids"].items()
            },
            "judged_boundary_sizes": {
                k: len(v) for k, v in agg["judged_boundary_paragraph_ids"].items()
            },
            "non_boundary_negative_paragraph_ids": sort_ids(agg["non_boundary_negative_paragraph_ids"]),
            "non_boundary_negative_count": len(agg["non_boundary_negative_paragraph_ids"]),
            "unjudged_paragraph_ids": sort_ids(agg["unjudged_paragraph_ids"]),
            "unjudged_count": len(agg["unjudged_paragraph_ids"]),
        }
        candidate_rows.append(row)

    write_jsonl(candidate_output_path, candidate_rows)


if __name__ == "__main__":
    domains = [
        "electrochemistry",
        "cell_culture",
        "immunotechnology",
        "inorganic_materials",
        "organic_synthesis",
        "polymer_materials",
    ]

    for d in domains:
        input_path = f"data/{d}/stage1_relevance/judge_results_stage1_gpt-5.4.jsonl"
        paragraph_output_path = f"data/{d}/stage1_relevance/paragraph_labels_from_judge.jsonl"
        candidate_output_path = f"data/{d}/stage1_relevance/candidate_summary_from_judge.jsonl"

        relabel_from_judge(
            input_path=input_path,
            paragraph_output_path=paragraph_output_path,
            candidate_output_path=candidate_output_path,
        )

        print("done")

# python scripts/stage1_relevance/check_boundary_paragraph.py
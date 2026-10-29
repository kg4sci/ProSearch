import json
from pathlib import Path
from collections import Counter


# =========================
# IO
# =========================
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


# =========================
# Helpers
# =========================
def in_range(x, lower=None, upper=None):
    if lower is not None and x < lower:
        return False
    if upper is not None and x > upper:
        return False
    return True


def is_exploration(candidate):
    explore_field = candidate.get("explore_field")
    if explore_field not in (None, "", []):
        return True
    t = str(candidate.get("type", ""))
    return "exploration" in t


def is_negation(candidate):
    return bool(candidate.get("has_negation", False))


def classify_candidate(candidate):
    neg = is_negation(candidate)
    exp = is_exploration(candidate)

    if neg and exp:
        return "neg_exploration"
    if neg:
        return "negation"
    if exp:
        return "exploration"
    return "positive"


def sorted_counter_dict(counter_obj):
    def sort_key(item):
        k = item[0]
        try:
            return (0, int(k))
        except Exception:
            return (1, str(k))

    return dict(sorted(counter_obj.items(), key=sort_key))


def nested_inc(d, *keys):
    cur = d
    for k in keys[:-1]:
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = cur.get(keys[-1], 0) + 1


def get_support(c):
    return c.get("support", c.get("exact_support_count", 0))


def get_logic_nm(c):
    return c.get("logic_near_miss_1_count", 0)


def get_logic_mm(c):
    return c.get("logic_near_miss_mismatch_1_count", 0)


def get_answer_missing(c):
    return c.get("answer_missing_count", 0)


def get_hard_conflict(c):
    return c.get("neg_hard_conflict_count", 0)


def get_distinct_explore_values(c):
    return c.get("distinct_explored_value_count", None)


# =========================
# Filter Logic
# =========================
def pass_positive_rules(c, cfg):
    logic_nm = get_logic_nm(c)
    logic_mm = get_logic_mm(c)

    if not in_range(logic_nm, cfg["logic_near_miss_min"], cfg["logic_near_miss_max"]):
        return False, "logic_near_miss_out_of_range"

    if not in_range(logic_mm, cfg["logic_mismatch_min"], cfg["logic_mismatch_max"]):
        return False, "logic_mismatch_out_of_range"

    return True, "pass"


def pass_negation_rules(c, cfg):
    hard_conflict = get_hard_conflict(c)
    logic_nm = get_logic_nm(c)
    logic_mm = get_logic_mm(c)

    if not in_range(hard_conflict, cfg["neg_hard_conflict_min"], cfg["neg_hard_conflict_max"]):
        return False, "neg_hard_conflict_out_of_range"

    if cfg.get("use_logic_near_miss_for_negation", False):
        if not in_range(logic_nm, cfg["logic_near_miss_min"], cfg["logic_near_miss_max"]):
            return False, "logic_near_miss_out_of_range"
        if not in_range(logic_mm, cfg["logic_mismatch_min"], cfg["logic_mismatch_max"]):
            return False, "logic_mismatch_out_of_range"

    return True, "pass"


def pass_exploration_rules(c, cfg):
    distinct_explore_values = get_distinct_explore_values(c)
    if cfg.get("distinct_explored_value_min") is not None and distinct_explore_values is not None:
        if distinct_explore_values < cfg["distinct_explored_value_min"]:
            return False, "distinct_explored_value_too_small"

    logic_nm = get_logic_nm(c)
    logic_mm = get_logic_mm(c)
    answer_missing = get_answer_missing(c)

    logic_ok = (
        in_range(logic_nm, cfg["logic_near_miss_min"], cfg["logic_near_miss_max"])
        and in_range(logic_mm, cfg["logic_mismatch_min"], cfg["logic_mismatch_max"])
    )
    answer_missing_ok = in_range(
        answer_missing,
        cfg["answer_missing_min"],
        cfg["answer_missing_max"],
    )

    mode = cfg.get("exploration_boundary_mode", "or")

    if mode == "or":
        ok = logic_ok or answer_missing_ok
    elif mode == "and":
        ok = logic_ok and answer_missing_ok
    elif mode == "logic_only":
        ok = logic_ok
    elif mode == "answer_only":
        ok = answer_missing_ok
    else:
        raise ValueError(f"Unsupported exploration_boundary_mode: {mode}")

    if not ok:
        return False, "exploration_boundary_not_satisfied"

    return True, "pass"


def pass_neg_exploration_rules(c, cfg):
    distinct_explore_values = get_distinct_explore_values(c)
    if cfg.get("distinct_explored_value_min") is not None and distinct_explore_values is not None:
        if distinct_explore_values < cfg["distinct_explored_value_min"]:
            return False, "distinct_explored_value_too_small"

    hard_conflict = get_hard_conflict(c)
    if not in_range(hard_conflict, cfg["neg_hard_conflict_min"], cfg["neg_hard_conflict_max"]):
        return False, "neg_hard_conflict_out_of_range"

    logic_nm = get_logic_nm(c)
    logic_mm = get_logic_mm(c)
    answer_missing = get_answer_missing(c)

    logic_ok = (
        in_range(logic_nm, cfg["logic_near_miss_min"], cfg["logic_near_miss_max"])
        and in_range(logic_mm, cfg["logic_mismatch_min"], cfg["logic_mismatch_max"])
    )
    answer_missing_ok = in_range(
        answer_missing,
        cfg["answer_missing_min"],
        cfg["answer_missing_max"],
    )

    mode = cfg.get("boundary_mode", "or")

    if mode == "or":
        ok = logic_ok or answer_missing_ok
    elif mode == "and":
        ok = logic_ok and answer_missing_ok
    elif mode == "logic_only":
        ok = logic_ok
    elif mode == "answer_only":
        ok = answer_missing_ok
    else:
        raise ValueError(f"Unsupported boundary_mode: {mode}")

    if not ok:
        return False, "neg_exploration_boundary_not_satisfied"

    return True, "pass"


def pass_candidate(c, cfg):
    kind = classify_candidate(c)

    support = get_support(c)
    if not in_range(support, cfg["support_min"], cfg["support_max"]):
        return False, "support_out_of_range", kind

    if kind == "positive":
        ok, reason = pass_positive_rules(c, cfg["positive"])
        return ok, reason, kind

    if kind == "negation":
        ok, reason = pass_negation_rules(c, cfg["negation"])
        return ok, reason, kind

    if kind == "exploration":
        ok, reason = pass_exploration_rules(c, cfg["exploration"])
        return ok, reason, kind

    if kind == "neg_exploration":
        ok, reason = pass_neg_exploration_rules(c, cfg["neg_exploration"])
        return ok, reason, kind

    return False, "unknown_kind", kind


# =========================
# Stats
# =========================
def update_counts(container, row):
    kind = classify_candidate(row)
    t = row.get("type")
    tid = row.get("type_id")
    neg = str(bool(row.get("has_negation", False)))
    numeric = str(bool(row.get("has_numeric_constraint", False)))
    range_flag = str(bool(row.get("has_range_constraint", False)))
    source = row.get("candidate_source", "unknown")
    support = str(get_support(row))

    container["kind_counts"][kind] += 1
    container["type_counts"][t] += 1
    container["type_id_counts"][tid] += 1
    container["negation_counts"][neg] += 1
    container["numeric_counts"][numeric] += 1
    container["range_counts"][range_flag] += 1
    container["source_counts"][source] += 1
    container["support_counts"][support] += 1

    nested_inc(container["type_by_negation_counts"], t, neg)
    nested_inc(container["type_by_numeric_counts"], t, numeric)
    nested_inc(container["type_by_range_counts"], t, range_flag)
    nested_inc(container["type_id_by_negation_counts"], tid, neg)
    nested_inc(container["kind_by_negation_counts"], kind, neg)
    nested_inc(container["kind_by_numeric_counts"], kind, numeric)
    nested_inc(container["kind_by_range_counts"], kind, range_flag)


def init_stats_block():
    return {
        "kind_counts": Counter(),
        "type_counts": Counter(),
        "type_id_counts": Counter(),
        "negation_counts": Counter(),
        "numeric_counts": Counter(),
        "range_counts": Counter(),
        "source_counts": Counter(),
        "support_counts": Counter(),
        "type_by_negation_counts": {},
        "type_by_numeric_counts": {},
        "type_by_range_counts": {},
        "type_id_by_negation_counts": {},
        "kind_by_negation_counts": {},
        "kind_by_numeric_counts": {},
        "kind_by_range_counts": {},
    }


def finalize_stats_block(block):
    return {
        "kind_counts": dict(block["kind_counts"]),
        "type_counts": dict(block["type_counts"]),
        "type_id_counts": dict(block["type_id_counts"]),
        "negation_counts": dict(block["negation_counts"]),
        "numeric_counts": dict(block["numeric_counts"]),
        "range_counts": dict(block["range_counts"]),
        "source_counts": dict(block["source_counts"]),
        "support_counts": sorted_counter_dict(block["support_counts"]),
        "type_by_negation_counts": block["type_by_negation_counts"],
        "type_by_numeric_counts": block["type_by_numeric_counts"],
        "type_by_range_counts": block["type_by_range_counts"],
        "type_id_by_negation_counts": block["type_id_by_negation_counts"],
        "kind_by_negation_counts": block["kind_by_negation_counts"],
        "kind_by_numeric_counts": block["kind_by_numeric_counts"],
        "kind_by_range_counts": block["kind_by_range_counts"],
    }


# =========================
# Main
# =========================
def filter_boundary_shortlist(
    input_path,
    output_path,
    stats_output_path,
    cfg,
):
    rows = list(read_jsonl(input_path))

    kept = []
    reject_reason_counts = Counter()

    raw_stats = init_stats_block()
    kept_stats = init_stats_block()

    for row in rows:
        update_counts(raw_stats, row)

        ok, reason, _kind = pass_candidate(row, cfg)
        if ok:
            kept.append(row)
            update_counts(kept_stats, row)
        else:
            reject_reason_counts[reason] += 1

    write_jsonl(output_path, kept)

    stats = {
        "total_input_candidates": len(rows),
        "total_kept_candidates": len(kept),
        "keep_ratio": (len(kept) / len(rows)) if rows else 0.0,
        "config": cfg,
        # "raw": finalize_stats_block(raw_stats),
        "kept": finalize_stats_block(kept_stats),
        "reject_reason_counts": dict(reject_reason_counts),
    }

    write_json(stats_output_path, stats)
    print(f"[OK] Saved filtered shortlist: {output_path}")
    print(f"[OK] Saved shortlist stats: {stats_output_path}")
    print(f"[OK] Kept {len(kept)} / {len(rows)} candidates")
    print(stats['kept']["kind_counts"])

if __name__ == "__main__":
    domains = [
        # "electrochemistry",
        "cell_culture",
        # "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        # "polymer_materials",
    ]

    filter_config = {
        "support_min": 3,
        "support_max": 8,

        # 正向复现
        "positive": {
            "logic_near_miss_min": 5,
            "logic_near_miss_max": 50,
            "logic_mismatch_min": 5,
            "logic_mismatch_max": 50,
        },

        # 否定复现
        "negation": {
            "neg_hard_conflict_min": 3,
            "neg_hard_conflict_max": 100,
            "use_logic_near_miss_for_negation": False,
            "logic_near_miss_min": 1,
            "logic_near_miss_max": 30,
            "logic_mismatch_min": 1,
            "logic_mismatch_max": 30,
        },

        # 正向探索
        "exploration": {
            "distinct_explored_value_min": 1,
            "logic_near_miss_min": 1,
            "logic_near_miss_max": 100,
            "logic_mismatch_min": 1,
            "logic_mismatch_max": 100,
            "answer_missing_min": 3,
            "answer_missing_max": 10,
            "exploration_boundary_mode": "or",
        },

        # 否定探索
        "neg_exploration": {
            "distinct_explored_value_min": 1,
            "neg_hard_conflict_min": 3,
            "neg_hard_conflict_max": 100,
            "logic_near_miss_min": 3,
            "logic_near_miss_max": 100,
            "logic_mismatch_min": 3,
            "logic_mismatch_max": 100,
            "answer_missing_min": 3,
            "answer_missing_max": 100,
            "boundary_mode": "or",
        },
    }

    for d in domains:
        print(f"\n=== Boundary shortlist filtering: {d} ===")
        input_path = f"seed_data/{d}/constraints/constraint_candidates_with_boundary_stats.jsonl"
        output_path = f"seed_data/{d}/constraints/constraint_candidates_boundary_shortlist.jsonl"
        stats_output_path = f"seed_data/{d}/constraints/constraint_candidates_boundary_shortlist.stats.json"

        filter_boundary_shortlist(
            input_path=input_path,
            output_path=output_path,
            stats_output_path=stats_output_path,
            cfg=filter_config,
        )


# python scripts/constraint_set/filter_near_miss_pool.py
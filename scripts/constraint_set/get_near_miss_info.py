import json
import os
import time
import multiprocessing as mp
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

EPS = 1e-8

G_SETUPS = None
G_FIELD_PRESENCE_IDX = None
G_FIELD_VALUE_IDX = None
G_ALL_REC_IDS = None
G_EMPTY_REC_IDS = frozenset()


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
# Normalization helpers
# =========================
def normalize_unit(unit):
    if unit is None:
        return None
    unit = str(unit).strip()
    if not unit:
        return None
    unit = unit.replace("µ", "μ")
    unit = unit.replace("ug", "μg").replace("mcg", "μg")
    return unit



def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    return []


# =========================
# Preprocessing / globals
# =========================
def _normalize_parsed(parsed):
    if not isinstance(parsed, dict):
        return parsed
    out = dict(parsed)
    out["_norm_unit"] = normalize_unit(out.get("unit"))
    return out



def _preprocess_item_bindings(item_bindings):
    out = []
    item_bindings = item_bindings or {}
    for group_name, items in item_bindings.items():
        items = items or []
        field_union = set()
        normalized_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized_item = {}
            for field, val in item.items():
                entries = ensure_list(val)
                new_entries = []
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    ee = dict(e)
                    if "parsed" in ee:
                        ee["parsed"] = _normalize_parsed(ee.get("parsed"))
                    new_entries.append(ee)
                normalized_item[field] = new_entries
                if new_entries:
                    field_union.add(field)
            if normalized_item:
                normalized_items.append(normalized_item)
        if field_union and normalized_items:
            out.append({
                "group_name": group_name,
                "field_union": frozenset(field_union),
                "items": normalized_items,
            })
    return out



def load_and_prepare_runtime(setup_records_path):
    global G_SETUPS, G_FIELD_PRESENCE_IDX, G_FIELD_VALUE_IDX, G_ALL_REC_IDS

    setups = []
    field_presence_idx = defaultdict(set)
    field_value_idx = defaultdict(lambda: defaultdict(set))

    for rec_idx, rec in enumerate(read_jsonl(setup_records_path)):
        flat_values = rec.get("flat_values", {}) or {}
        new_flat_values = {}

        for field, raw_entries in flat_values.items():
            entries = ensure_list(raw_entries)
            norm_entries = []
            for e in entries:
                if not isinstance(e, dict):
                    continue
                ee = dict(e)
                if "parsed" in ee:
                    ee["parsed"] = _normalize_parsed(ee.get("parsed"))
                norm_entries.append(ee)
            if not norm_entries:
                continue

            new_flat_values[field] = norm_entries
            field_presence_idx[field].add(rec_idx)
            for e in norm_entries:
                val = e.get("value")
                if val is None:
                    continue
                try:
                    field_value_idx[field][val].add(rec_idx)
                except TypeError:
                    # Skip unhashable values safely.
                    pass

        setups.append({
            "paragraph_id": str(rec.get("paragraph_id")),
            "flat_values": new_flat_values,
            "binding_groups": _preprocess_item_bindings(rec.get("item_bindings", {})),
        })

    G_SETUPS = setups
    G_FIELD_PRESENCE_IDX = {k: frozenset(v) for k, v in field_presence_idx.items()}
    G_FIELD_VALUE_IDX = {
        field: {value: frozenset(idx_set) for value, idx_set in value_map.items()}
        for field, value_map in field_value_idx.items()
    }
    G_ALL_REC_IDS = frozenset(range(len(setups)))



def _preprocess_literal(lit):
    lit = dict(lit)
    if "parsed" in lit:
        lit["parsed"] = _normalize_parsed(lit.get("parsed"))
    if "range" in lit and isinstance(lit.get("range"), dict):
        lit["range"] = dict(lit["range"])
        lit["range"]["_norm_unit"] = normalize_unit(lit["range"].get("unit"))
    return lit



def preprocess_candidates(candidates):
    out = []
    for cand in candidates:
        row = dict(cand)
        constraints = [_preprocess_literal(c) for c in (cand.get("constraints", []) or [])]
        negated_constraints = [c for c in constraints if c.get("op") == "!=" ]
        non_negated_constraints = [c for c in constraints if c.get("op") != "!=" ]
        explore_field = cand.get("explore_field")
        is_exploration = explore_field not in (None, "", [])
        has_negation = bool(cand.get("has_negation", False)) or bool(negated_constraints)

        row["_constraints"] = constraints
        row["_negated_constraints"] = negated_constraints
        row["_non_negated_constraints"] = non_negated_constraints
        row["_explore_field"] = explore_field
        row["_is_exploration"] = is_exploration
        row["_has_negation"] = has_negation
        row["_constraints_by_field"] = {c["field"]: c for c in constraints}
        row["_candidate_fields"] = frozenset(c["field"] for c in constraints)
        out.append(row)
    return out


# =========================
# Matching helpers
# =========================
def eq_number(a, b):
    return abs(float(a) - float(b)) <= EPS



def numeric_match(literal, entry):
    s_parsed = entry.get("parsed")
    if not s_parsed:
        return False

    s_unit = s_parsed.get("_norm_unit", normalize_unit(s_parsed.get("unit")))

    if literal["op"] == "=":
        q_parsed = literal.get("parsed")
        if not q_parsed:
            return False
        q_unit = q_parsed.get("_norm_unit", normalize_unit(q_parsed.get("unit")))
        if q_unit != s_unit:
            return False

        if s_parsed["kind"] == "point":
            return eq_number(s_parsed["number"], q_parsed["number"])
        if s_parsed["kind"] == "range":
            return s_parsed["lower"] - EPS <= q_parsed["number"] <= s_parsed["upper"] + EPS
        return False

    if literal["op"] == "between":
        qrange = literal.get("range", {})
        q_unit = qrange.get("_norm_unit", normalize_unit(qrange.get("unit")))
        if q_unit != s_unit:
            return False
        ql, qr = qrange["lower"], qrange["upper"]

        if s_parsed["kind"] == "point":
            return ql - EPS <= s_parsed["number"] <= qr + EPS
        if s_parsed["kind"] == "range":
            return ql - EPS <= s_parsed["lower"] and s_parsed["upper"] <= qr + EPS
        return False

    return False



def literal_matches_entries(literal, entries):
    if not entries:
        return False

    if literal["op"] == "!=":
        return all(e.get("value") != literal["value"] for e in entries)

    for e in entries:
        if literal["op"] == "=":
            if "parsed" in literal:
                if numeric_match(literal, e):
                    return True
            else:
                if e.get("value") == literal["value"]:
                    return True

        elif literal["op"] == "between":
            if numeric_match(literal, e):
                return True

    return False



def has_negated_value_conflict(literal, entries):
    if literal["op"] != "!=" or not entries:
        return False
    target = literal["value"]
    for e in entries:
        if e.get("value") == target:
            return True
    return False



def get_field_status(literal, flat_values):
    field = literal["field"]
    entries = flat_values.get(field, [])

    if not entries:
        return "missing"
    if literal_matches_entries(literal, entries):
        return "match"
    return "mismatch"


# =========================
# Candidate pool narrowing
# =========================
def get_anchor_rec_ids(literal):
    field = literal["field"]
    op = literal.get("op")

    # Strong anchor: exact categorical equality.
    if op == "=" and "parsed" not in literal:
        return G_FIELD_VALUE_IDX.get(field, {}).get(literal.get("value"), G_EMPTY_REC_IDS)

    # Weaker but safe anchor: field presence for numeric / range constraints.
    if op in {"=", "between"}:
        return G_FIELD_PRESENCE_IDX.get(field, G_EMPTY_REC_IDS)

    # Negated constraints are not safe anchors because "missing" can itself be a target boundary type.
    return None



def intersect_sets(sets_):
    if not sets_:
        return G_ALL_REC_IDS
    ordered = sorted(sets_, key=len)
    out = set(ordered[0])
    for s in ordered[1:]:
        out.intersection_update(s)
        if not out:
            break
    return out



def build_candidate_pool(candidate):
    """
    Safe superset of records that could contribute to any output bucket.

    Logic:
    - Exact support / explore_field_missing / negation buckets require all non-negated constraints to hold.
    - Fixed-constraint near-miss buckets allow one non-negated field to fail.

    So we build a union of intersections where we allow at most one non-negated anchor to drop.
    This preserves correctness while often cutting the scan set substantially.
    """
    anchors = []
    for lit in candidate["_non_negated_constraints"]:
        rec_ids = get_anchor_rec_ids(lit)
        if rec_ids is not None:
            anchors.append(rec_ids)

    if not anchors:
        return G_ALL_REC_IDS

    pool = set(intersect_sets(anchors))

    # Allow one non-negated anchor to fail for fixed_constraint_* near-miss buckets.
    for i in range(len(anchors)):
        other_sets = anchors[:i] + anchors[i + 1 :]
        pool.update(intersect_sets(other_sets))

    return pool


# =========================
# item_bindings helpers
# =========================
def get_item_field_entries(item, field):
    val = item.get(field, [])
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return [val]
    return []



def apply_same_item_adjustment(status_map, candidate, setup):
    candidate_fields = candidate["_candidate_fields"]
    constraints_by_field = candidate["_constraints_by_field"]

    for group in setup.get("binding_groups", []):
        overlap_fields = candidate_fields.intersection(group["field_union"])
        if len(overlap_fields) < 2:
            continue

        if not all(status_map.get(f) == "match" for f in overlap_fields):
            continue

        same_item_ok = False
        for item in group["items"]:
            ok = True
            for f in overlap_fields:
                lit = constraints_by_field[f]
                entries = get_item_field_entries(item, f)
                if not literal_matches_entries(lit, entries):
                    ok = False
                    break
            if ok:
                same_item_ok = True
                break

        if not same_item_ok:
            for f in overlap_fields:
                status_map[f] = "mismatch"

    return status_map


# =========================
# Candidate analysis
# =========================
def compute_status_map(candidate, rec):
    flat_values = rec.get("flat_values", {})
    status_map = {}
    for lit in candidate["_constraints"]:
        status_map[lit["field"]] = get_field_status(lit, flat_values)
    status_map = apply_same_item_adjustment(status_map, candidate, rec)
    return status_map



def _remove_exact_support_from_field_sets(field_sets, exact_support_pids):
    for field, pid_set in field_sets.items():
        pid_set -= exact_support_pids



def analyze_candidate(candidate):
    constraints = candidate["_constraints"]
    negated_constraints = candidate["_negated_constraints"]
    non_negated_constraints = candidate["_non_negated_constraints"]
    explore_field = candidate["_explore_field"]
    is_exploration = candidate["_is_exploration"]
    has_negation = candidate["_has_negation"]

    exact_support_pids = set()
    explore_field_missing_pids = set()

    fixed_constraint_near_miss_pids = set()
    fixed_constraint_missing_1_pids = set()
    fixed_constraint_mismatch_1_pids = set()

    fixed_constraint_near_miss_by_field_sets = defaultdict(set)
    fixed_constraint_missing_1_by_field_sets = defaultdict(set)
    fixed_constraint_mismatch_1_by_field_sets = defaultdict(set)

    negated_field_mismatch_pids = set()
    negated_field_mismatch_by_field_sets = defaultdict(set)

    negated_field_missing_pids = set()
    negated_field_missing_by_field_sets = defaultdict(set)

    candidate_rec_ids = build_candidate_pool(candidate)

    for rec_idx in candidate_rec_ids:
        rec = G_SETUPS[rec_idx]
        pid = rec["paragraph_id"]
        status_map = compute_status_map(candidate, rec)
        all_constraints_match = all(status_map.get(lit["field"]) == "match" for lit in constraints)

        if all_constraints_match:
            if is_exploration:
                entries = rec.get("flat_values", {}).get(explore_field, [])
                if entries:
                    exact_support_pids.add(pid)
                else:
                    explore_field_missing_pids.add(pid)
            else:
                exact_support_pids.add(pid)
            continue

        if has_negation:
            for lit in negated_constraints:
                other_ok = True
                for other in constraints:
                    if other["field"] == lit["field"]:
                        continue
                    if status_map.get(other["field"]) != "match":
                        other_ok = False
                        break

                if not other_ok:
                    continue

                entries = rec.get("flat_values", {}).get(lit["field"], [])
                if has_negated_value_conflict(lit, entries):
                    negated_field_mismatch_pids.add(pid)
                    negated_field_mismatch_by_field_sets[lit["field"]].add(pid)

            for lit in negated_constraints:
                other_ok = True
                for other in constraints:
                    if other["field"] == lit["field"]:
                        continue
                    if status_map.get(other["field"]) != "match":
                        other_ok = False
                        break

                if not other_ok:
                    continue

                if status_map.get(lit["field"]) == "missing":
                    negated_field_missing_pids.add(pid)
                    negated_field_missing_by_field_sets[lit["field"]].add(pid)

            if all(status_map.get(lit["field"]) == "match" for lit in negated_constraints):
                missing_fields = [
                    lit["field"]
                    for lit in non_negated_constraints
                    if status_map.get(lit["field"]) == "missing"
                ]
                mismatch_fields = [
                    lit["field"]
                    for lit in non_negated_constraints
                    if status_map.get(lit["field"]) == "mismatch"
                ]

                if len(missing_fields) + len(mismatch_fields) == 1:
                    fixed_constraint_near_miss_pids.add(pid)

                    if len(missing_fields) == 1:
                        f = missing_fields[0]
                        fixed_constraint_missing_1_pids.add(pid)
                        fixed_constraint_near_miss_by_field_sets[f].add(pid)
                        fixed_constraint_missing_1_by_field_sets[f].add(pid)

                    elif len(mismatch_fields) == 1:
                        f = mismatch_fields[0]
                        fixed_constraint_mismatch_1_pids.add(pid)
                        fixed_constraint_near_miss_by_field_sets[f].add(pid)
                        fixed_constraint_mismatch_1_by_field_sets[f].add(pid)

        else:
            missing_fields = [lit["field"] for lit in constraints if status_map.get(lit["field"]) == "missing"]
            mismatch_fields = [lit["field"] for lit in constraints if status_map.get(lit["field"]) == "mismatch"]

            if len(missing_fields) + len(mismatch_fields) == 1:
                fixed_constraint_near_miss_pids.add(pid)

                if len(missing_fields) == 1:
                    f = missing_fields[0]
                    fixed_constraint_missing_1_pids.add(pid)
                    fixed_constraint_near_miss_by_field_sets[f].add(pid)
                    fixed_constraint_missing_1_by_field_sets[f].add(pid)

                elif len(mismatch_fields) == 1:
                    f = mismatch_fields[0]
                    fixed_constraint_mismatch_1_pids.add(pid)
                    fixed_constraint_near_miss_by_field_sets[f].add(pid)
                    fixed_constraint_mismatch_1_by_field_sets[f].add(pid)

    explore_field_missing_pids -= exact_support_pids
    fixed_constraint_near_miss_pids -= exact_support_pids
    fixed_constraint_missing_1_pids -= exact_support_pids
    fixed_constraint_mismatch_1_pids -= exact_support_pids
    negated_field_mismatch_pids -= exact_support_pids
    negated_field_missing_pids -= exact_support_pids

    _remove_exact_support_from_field_sets(fixed_constraint_near_miss_by_field_sets, exact_support_pids)
    _remove_exact_support_from_field_sets(fixed_constraint_missing_1_by_field_sets, exact_support_pids)
    _remove_exact_support_from_field_sets(fixed_constraint_mismatch_1_by_field_sets, exact_support_pids)
    _remove_exact_support_from_field_sets(negated_field_mismatch_by_field_sets, exact_support_pids)
    _remove_exact_support_from_field_sets(negated_field_missing_by_field_sets, exact_support_pids)

    explore_field_missing_ratio = None
    if is_exploration:
        denom = len(exact_support_pids) + len(explore_field_missing_pids)
        explore_field_missing_ratio = (len(explore_field_missing_pids) / denom) if denom > 0 else None

    out = {k: v for k, v in candidate.items() if not str(k).startswith("_")}

    out["exact_support_count"] = len(exact_support_pids)

    out["fixed_constraint_near_miss_1_count"] = len(fixed_constraint_near_miss_pids)
    out["fixed_constraint_missing_1_count"] = len(fixed_constraint_missing_1_pids)
    out["fixed_constraint_mismatch_1_count"] = len(fixed_constraint_mismatch_1_pids)

    out["fixed_constraint_near_miss_by_field"] = {
        k: len(v) for k, v in fixed_constraint_near_miss_by_field_sets.items()
    }
    out["fixed_constraint_missing_1_by_field"] = {
        k: len(v) for k, v in fixed_constraint_missing_1_by_field_sets.items()
    }
    out["fixed_constraint_mismatch_1_by_field"] = {
        k: len(v) for k, v in fixed_constraint_mismatch_1_by_field_sets.items()
    }

    out["negated_field_mismatch_count"] = len(negated_field_mismatch_pids)
    out["negated_field_mismatch_by_field"] = {
        k: len(v) for k, v in negated_field_mismatch_by_field_sets.items()
    }

    out["negated_field_missing_count"] = len(negated_field_missing_pids)
    out["negated_field_missing_by_field"] = {
        k: len(v) for k, v in negated_field_missing_by_field_sets.items()
    }

    out["explore_field_missing_count"] = len(explore_field_missing_pids)
    out["explore_field_missing_ratio"] = explore_field_missing_ratio

    out["candidate_pool_size"] = len(candidate_rec_ids)
    return out



def process_chunk(chunk):
    return [analyze_candidate(c) for c in chunk]


# =========================
# Main
# =========================
def compute_with_pool(candidates, max_workers=1, chunk_size=50):
    chunks = [candidates[i : i + chunk_size] for i in range(0, len(candidates), chunk_size)]
    all_rows = []

    # Use fork-based multiprocessing only when it is available.
    # This lets workers share the already-loaded runtime via copy-on-write instead of re-reading it.
    can_fork = hasattr(mp, "get_context") and os.name != "nt"
    use_mp = max_workers and max_workers > 1 and can_fork

    if use_mp:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futures = [ex.submit(process_chunk, chunk) for chunk in chunks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Computing boundary stats"):
                all_rows.extend(fut.result())
    else:
        if max_workers and max_workers > 1 and not can_fork:
            print("[WARN] fork is unavailable on this platform; falling back to single-process mode.")
        for chunk in tqdm(chunks, total=len(chunks), desc="Computing boundary stats"):
            all_rows.extend(process_chunk(chunk))

    return all_rows



def main(
    setup_records_path,
    merged_candidates_path,
    output_path,
    stats_output_path,
    max_workers=1,
    chunk_size=50,
):
    t0 = time.time()

    print(f"[INFO] Loading runtime from: {setup_records_path}")
    load_and_prepare_runtime(setup_records_path)
    t1 = time.time()

    candidates = list(read_jsonl(merged_candidates_path))
    if not candidates:
        print(f"[WARN] No candidates found: {merged_candidates_path}")
        return
    candidates = preprocess_candidates(candidates)
    t2 = time.time()

    all_rows = compute_with_pool(candidates, max_workers=max_workers, chunk_size=chunk_size)
    t3 = time.time()

    all_rows.sort(key=lambda x: str(x.get("candidate_id", "")))
    write_jsonl(output_path, all_rows)

    total_records = len(G_SETUPS)
    pool_sizes = [r.get("candidate_pool_size", total_records) for r in all_rows]

    stats = {
        "total_candidates": len(all_rows),
        "total_setup_records": total_records,
        "avg_candidate_pool_size": round(sum(pool_sizes) / len(pool_sizes), 4) if pool_sizes else None,
        "median_candidate_pool_size": sorted(pool_sizes)[len(pool_sizes) // 2] if pool_sizes else None,
        "avg_pool_ratio": round((sum(pool_sizes) / len(pool_sizes)) / total_records, 6) if pool_sizes and total_records else None,
        "source_counts": dict(Counter(r.get("candidate_source", "unknown") for r in all_rows)),
        "negation_counts": dict(Counter(str(bool(r.get("has_negation", False))) for r in all_rows)),
        "type_counts": dict(Counter(r.get("type") for r in all_rows)),
        "type_id_counts": dict(Counter(r.get("type_id") for r in all_rows)),
        "exact_support_counts": dict(sorted(Counter(str(r.get("exact_support_count", 0)) for r in all_rows).items(), key=lambda x: int(x[0]))),
        "fixed_constraint_near_miss_1_counts": dict(sorted(Counter(str(r.get("fixed_constraint_near_miss_1_count", 0)) for r in all_rows).items(), key=lambda x: int(x[0]))),
        "fixed_constraint_missing_1_counts": dict(sorted(Counter(str(r.get("fixed_constraint_missing_1_count", 0)) for r in all_rows).items(), key=lambda x: int(x[0]))),
        "fixed_constraint_mismatch_1_counts": dict(sorted(Counter(str(r.get("fixed_constraint_mismatch_1_count", 0)) for r in all_rows).items(), key=lambda x: int(x[0]))),
        "negated_field_mismatch_counts": dict(sorted(Counter(str(r.get("negated_field_mismatch_count", 0)) for r in all_rows).items(), key=lambda x: int(x[0]))),
        "negated_field_missing_counts": dict(sorted(Counter(str(r.get("negated_field_missing_count", 0)) for r in all_rows).items(), key=lambda x: int(x[0]))),
        "explore_field_missing_counts": dict(sorted(Counter(str(r.get("explore_field_missing_count", 0)) for r in all_rows).items(), key=lambda x: int(x[0]))),
        "candidates_with_any_fixed_constraint_near_miss": sum(1 for r in all_rows if r.get("fixed_constraint_near_miss_1_count", 0) > 0),
        "candidates_with_any_negated_field_mismatch": sum(1 for r in all_rows if r.get("negated_field_mismatch_count", 0) > 0),
        "candidates_with_any_negated_field_missing": sum(1 for r in all_rows if r.get("negated_field_missing_count", 0) > 0),
        "candidates_with_any_explore_field_missing": sum(1 for r in all_rows if r.get("explore_field_missing_count", 0) > 0),
        "timing_seconds": {
            "load_runtime": round(t1 - t0, 4),
            "load_candidates": round(t2 - t1, 4),
            "compute": round(t3 - t2, 4),
            "total": round(t3 - t0, 4),
        },
    }
    write_json(stats_output_path, stats)

    print(f"[OK] Saved boundary stats candidates: {output_path}")
    print(f"[OK] Saved boundary stats summary: {stats_output_path}")
    print(f"[OK] Total processed candidates: {len(all_rows)}")
    print(f"[OK] Total setup records: {total_records}")
    if pool_sizes:
        print(f"[OK] Avg candidate pool size: {stats['avg_candidate_pool_size']} / {total_records}")


if __name__ == "__main__":
    domains = [
        # "electrochemistry",
        # "cell_culture",
        "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        # "polymer_materials",
    ]


    max_workers = 32
    chunk_size = 50

    for d in domains:
        print(f"\n=== Boundary stats stage: {d} ===")
        setup_records_path = f"seed_data/{d}/constraints/setup_records.jsonl"
        merged_candidates_path = f"seed_data/{d}/constraints/constraint_candidates_merged.jsonl"

        output_path = f"seed_data/{d}/constraints/constraint_candidates_with_boundary_stats.jsonl"
        stats_output_path = f"seed_data/{d}/constraints/constraint_candidates_with_boundary_stats.stats.json"

        main(
            setup_records_path=setup_records_path,
            merged_candidates_path=merged_candidates_path,
            output_path=output_path,
            stats_output_path=stats_output_path,
            max_workers=max_workers,
            chunk_size=chunk_size,
        )



# conda activate retrieve_dataset
# python scripts/constraint_set/get_near_miss_info.py


import json
import os
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

EPS = 1e-8
G_SETUPS = None


# =========================
# IO
# =========================
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


# =========================
# Globals / init
# =========================
def init_worker(setup_records_path):
    global G_SETUPS
    G_SETUPS = list(read_jsonl(setup_records_path))


# =========================
# Basic helpers
# =========================
def unique_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================
# Matching helpers
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


def eq_number(a, b):
    return abs(float(a) - float(b)) <= EPS


def numeric_match(literal, entry):
    s_parsed = entry.get("parsed")
    if not s_parsed:
        return False

    if literal["op"] == "=":
        q_parsed = literal.get("parsed")
        if not q_parsed:
            return False
        if normalize_unit(q_parsed.get("unit")) != normalize_unit(s_parsed.get("unit")):
            return False

        if s_parsed["kind"] == "point":
            return eq_number(s_parsed["number"], q_parsed["number"])
        if s_parsed["kind"] == "range":
            return s_parsed["lower"] - EPS <= q_parsed["number"] <= s_parsed["upper"] + EPS
        return False

    if literal["op"] == "between":
        qrange = literal.get("range", {})
        if normalize_unit(qrange.get("unit")) != normalize_unit(s_parsed.get("unit")):
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
# item_bindings helpers
# =========================
def get_candidate_binding_groups(candidate, setup):
    item_bindings = setup.get("item_bindings", {}) or {}
    candidate_fields = {c["field"] for c in candidate.get("constraints", [])}

    groups = []
    for group_name, items in item_bindings.items():
        if not items:
            continue
        field_union = set()
        for item in items:
            if isinstance(item, dict):
                field_union.update(item.keys())
        overlap = candidate_fields.intersection(field_union)
        if len(overlap) >= 2:
            groups.append((group_name, overlap))
    return groups


def get_item_field_entries(item, field):
    val = item.get(field, [])
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return [val]
    return []


def apply_same_item_adjustment(status_map, candidate, setup):
    item_bindings = setup.get("item_bindings", {}) or {}
    constraints_by_field = {c["field"]: c for c in candidate.get("constraints", [])}

    for group_name, overlap_fields in get_candidate_binding_groups(candidate, setup):
        items = item_bindings.get(group_name, [])
        if not items:
            continue

        if not all(status_map.get(f) == "match" for f in overlap_fields):
            continue

        same_item_ok = False
        for item in items:
            if not isinstance(item, dict):
                continue
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
def split_constraints(candidate):
    constraints = candidate.get("constraints", []) or []
    negated_constraints = [c for c in constraints if c.get("op") == "!="]
    non_negated_constraints = [c for c in constraints if c.get("op") != "!="]
    explore_field = candidate.get("explore_field")
    is_exploration = explore_field not in (None, "", [])
    has_negation = bool(candidate.get("has_negation", False)) or bool(negated_constraints)
    return constraints, negated_constraints, non_negated_constraints, explore_field, is_exploration, has_negation


def compute_status_map(candidate, rec):
    flat_values = rec.get("flat_values", {}) or {}
    status_map = {}
    for lit in candidate.get("constraints", []):
        status_map[lit["field"]] = get_field_status(lit, flat_values)
    status_map = apply_same_item_adjustment(status_map, candidate, rec)
    return status_map


def analyze_candidate_full_boundary(candidate):
    constraints, negated_constraints, non_negated_constraints, explore_field, is_exploration, has_negation = split_constraints(candidate)
    candidate_id = str(candidate["candidate_id"])

    # 正样本段落：直接以 candidate 自带 support_paragraph_ids 为准
    exact_support_paragraph_ids = unique_keep_order(candidate.get("support_paragraph_ids", []))
    exact_support_set = set(exact_support_paragraph_ids)

    fixed_constraint_missing_pids = set()
    fixed_constraint_mismatch_pids = set()
    negated_field_mismatch_pids = set()
    negated_field_missing_pids = set()
    explore_field_missing_pids = set()

    for rec in G_SETUPS:
        pid = str(rec.get("paragraph_id"))

        # 任何正样本段落都不再进入任何 boundary bucket
        if pid in exact_support_set:
            continue

        flat_values = rec.get("flat_values", {}) or {}
        status_map = compute_status_map(candidate, rec)
        all_constraints_match = all(status_map.get(lit["field"]) == "match" for lit in constraints)

        # exploration: 固定约束都满足，但 explore_field 缺失
        if is_exploration and all_constraints_match:
            explore_entries = flat_values.get(explore_field, [])
            if not explore_entries:
                explore_field_missing_pids.add(pid)

        if has_negation:
            # negated_field_mismatch
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

                entries = flat_values.get(lit["field"], [])
                if has_negated_value_conflict(lit, entries):
                    negated_field_mismatch_pids.add(pid)

            # negated_field_missing
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

            # fixed constraint near miss
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
                    if len(missing_fields) == 1:
                        fixed_constraint_missing_pids.add(pid)
                    elif len(mismatch_fields) == 1:
                        fixed_constraint_mismatch_pids.add(pid)

        else:
            missing_fields = [
                lit["field"]
                for lit in constraints
                if status_map.get(lit["field"]) == "missing"
            ]
            mismatch_fields = [
                lit["field"]
                for lit in constraints
                if status_map.get(lit["field"]) == "mismatch"
            ]

            if len(missing_fields) + len(mismatch_fields) == 1:
                if len(missing_fields) == 1:
                    fixed_constraint_missing_pids.add(pid)
                elif len(mismatch_fields) == 1:
                    fixed_constraint_mismatch_pids.add(pid)

    # 再兜底一次，确保所有 boundary 集合都不含正样本段落
    fixed_constraint_missing_pids -= exact_support_set
    fixed_constraint_mismatch_pids -= exact_support_set
    negated_field_mismatch_pids -= exact_support_set
    negated_field_missing_pids -= exact_support_set
    explore_field_missing_pids -= exact_support_set

    boundary_dict = {
        "fixed_constraint_missing_1": sorted(fixed_constraint_missing_pids),
        "fixed_constraint_mismatch_1": sorted(fixed_constraint_mismatch_pids),
        "negated_field_mismatch": sorted(negated_field_mismatch_pids),
        "negated_field_missing": sorted(negated_field_missing_pids),
        "explore_field_missing": sorted(explore_field_missing_pids),
    }

    return candidate_id, boundary_dict


def process_chunk(chunk):
    out = []
    for c in chunk:
        out.append(analyze_candidate_full_boundary(c))
    return out


# =========================
# Main
# =========================
def main(
    setup_records_path,
    sampled_constraints_path,
    output_path,
    max_workers=None,
    chunk_size=50,
):
    candidates = list(read_jsonl(sampled_constraints_path))
    if not candidates:
        print(f"[WARN] No candidates found: {sampled_constraints_path}")
        return

    max_workers = max_workers or max(1, mp.cpu_count() - 1)
    chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]

    result = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(setup_records_path,),
    ) as ex:
        futures = [ex.submit(process_chunk, chunk) for chunk in chunks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building full boundary dict"):
            rows = fut.result()
            for candidate_id, boundary_dict in rows:
                result[candidate_id] = boundary_dict

    result = {k: result[k] for k in sorted(result.keys())}
    write_json(output_path, result)

    print(f"[OK] Saved full boundary dict: {output_path}")
    print(f"[OK] Total candidates: {len(result)}")


if __name__ == "__main__":
    domains = [
        # "electrochemistry",
        # "cell_culture",
        # "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        "polymer_materials",
    ]

    max_workers = min(32, os.cpu_count() or 1, 64)
    chunk_size = 50

    for d in domains:
        print(f"\n=== Build full boundary dict: {d} ===")

        setup_records_path = f"seed_data/{d}/constraints/setup_records.jsonl"
        sampled_constraints_path = f"seed_data/{d}/constraints/sampled_constraints_stage1.jsonl"
        output_path = f"data/{d}/stage1_relevance/sampled_constraints_stage1.full_boundary_dict.json"

        main(
            setup_records_path=setup_records_path,
            sampled_constraints_path=sampled_constraints_path,
            output_path=output_path,
            max_workers=max_workers,
            chunk_size=chunk_size,
        )



# python scripts/stage1_relevance/get_all_boundary_paragraph.py

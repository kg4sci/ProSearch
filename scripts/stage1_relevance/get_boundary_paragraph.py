import json
import os
import random
import hashlib
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean, median

from tqdm import tqdm

EPS = 1e-8
G_SETUPS = None


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


def deterministic_sample(ids, sample_size, seed_text):
    ids = sorted(str(x) for x in set(ids))
    if len(ids) <= sample_size:
        return ids

    seed_int = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed_int)
    chosen = rng.sample(ids, sample_size)
    chosen.sort()
    return chosen


def summary_stats(values):
    values = list(values)
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "min": min(values),
        "max": max(values),
    }


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
    flat_values = rec.get("flat_values", {})
    status_map = {}
    for lit in candidate.get("constraints", []):
        status_map[lit["field"]] = get_field_status(lit, flat_values)
    status_map = apply_same_item_adjustment(status_map, candidate, rec)
    return status_map


def analyze_candidate_and_sample(candidate, sample_size_per_error=20, base_seed=42):
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

        # 任何已属于正样本的段落，都不再进入任何 boundary bucket
        if pid in exact_support_set:
            continue

        flat_values = rec.get("flat_values", {}) or {}
        status_map = compute_status_map(candidate, rec)
        all_constraints_match = all(status_map.get(lit["field"]) == "match" for lit in constraints)

        # exploration 专属：固定约束都满足，但 explore_field 缺失
        if is_exploration and all_constraints_match:
            explore_entries = flat_values.get(explore_field, [])
            if not explore_entries:
                explore_field_missing_pids.add(pid)

        if has_negation:
            # 1) negated_field_mismatch:
            # 其余约束都 match，某个 negated field 明确等于被禁止值
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

            # 2) negated_field_missing:
            # 其余约束都 match，negated field 缺失
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

            # 3) fixed constraint near miss:
            # 所有 negated constraints 都 match，且 non-negated constraints 中恰好 1 个 missing/mismatch
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
            # 无 negation 时：固定约束中恰好 1 个 missing/mismatch
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

    # 再次确保所有 boundary 集合都不含正样本段落
    fixed_constraint_missing_pids -= exact_support_set
    fixed_constraint_mismatch_pids -= exact_support_set
    negated_field_mismatch_pids -= exact_support_set
    negated_field_missing_pids -= exact_support_set
    explore_field_missing_pids -= exact_support_set

    sampled_boundary_paragraph_ids = {
        "fixed_constraint_missing_1": deterministic_sample(
            fixed_constraint_missing_pids,
            sample_size_per_error,
            f"{candidate_id}|fixed_constraint_missing_1|{base_seed}",
        ),
        "fixed_constraint_mismatch_1": deterministic_sample(
            fixed_constraint_mismatch_pids,
            sample_size_per_error,
            f"{candidate_id}|fixed_constraint_mismatch_1|{base_seed}",
        ),
        "negated_field_mismatch": deterministic_sample(
            negated_field_mismatch_pids,
            sample_size_per_error,
            f"{candidate_id}|negated_field_mismatch|{base_seed}",
        ),
        "negated_field_missing": deterministic_sample(
            negated_field_missing_pids,
            sample_size_per_error,
            f"{candidate_id}|negated_field_missing|{base_seed}",
        ),
        "explore_field_missing": deterministic_sample(
            explore_field_missing_pids,
            sample_size_per_error,
            f"{candidate_id}|explore_field_missing|{base_seed}",
        ),
    }

    boundary_pool_sizes = {
        "fixed_constraint_missing_1": len(fixed_constraint_missing_pids),
        "fixed_constraint_mismatch_1": len(fixed_constraint_mismatch_pids),
        "negated_field_mismatch": len(negated_field_mismatch_pids),
        "negated_field_missing": len(negated_field_missing_pids),
        "explore_field_missing": len(explore_field_missing_pids),
    }

    sampled_boundary_sizes = {k: len(v) for k, v in sampled_boundary_paragraph_ids.items()}

    final_candidate_paragraph_ids = []
    final_candidate_paragraph_ids.extend(exact_support_paragraph_ids)

    boundary_order = [
        "fixed_constraint_missing_1",
        "fixed_constraint_mismatch_1",
        "negated_field_mismatch",
        "negated_field_missing",
        "explore_field_missing",
    ]
    for err_type in boundary_order:
        final_candidate_paragraph_ids.extend(sampled_boundary_paragraph_ids[err_type])

    final_candidate_paragraph_ids = unique_keep_order(final_candidate_paragraph_ids)

    support_overlap_count_by_error = {
        "fixed_constraint_missing_1": len(fixed_constraint_missing_pids & exact_support_set),
        "fixed_constraint_mismatch_1": len(fixed_constraint_mismatch_pids & exact_support_set),
        "negated_field_mismatch": len(negated_field_mismatch_pids & exact_support_set),
        "negated_field_missing": len(negated_field_missing_pids & exact_support_set),
        "explore_field_missing": len(explore_field_missing_pids & exact_support_set),
    }

    explore_field_missing_support_overlap_count = support_overlap_count_by_error["explore_field_missing"]

    manifest_row = {
        "candidate_id": candidate_id,
        "domain": candidate.get("domain"),
        "type": candidate.get("type"),
        "type_id": candidate.get("type_id"),
        "constraints": candidate.get("constraints", []),
        "explore_field": candidate.get("explore_field"),
        "has_negation": bool(candidate.get("has_negation", False)),
        "has_numeric_constraint": bool(candidate.get("has_numeric_constraint", False)),
        "has_range_constraint": bool(candidate.get("has_range_constraint", False)),
        "candidate_source": candidate.get("candidate_source"),
        "support_paragraph_ids": exact_support_paragraph_ids,
        "support_count": len(exact_support_paragraph_ids),
        "sampled_boundary_paragraph_ids": sampled_boundary_paragraph_ids,
        "boundary_pool_sizes": boundary_pool_sizes,
        "sampled_boundary_sizes": sampled_boundary_sizes,
        "explore_field_missing_support_overlap_count": explore_field_missing_support_overlap_count,
        "support_overlap_count_by_error": support_overlap_count_by_error,
    }

    boundary_sample_count_before_dedup = int(sum(sampled_boundary_sizes.values()))
    boundary_sample_count_after_dedup = max(
        0, len(final_candidate_paragraph_ids) - len(exact_support_paragraph_ids)
    )
    boundary_overlap_count = max(
        0, boundary_sample_count_before_dedup - boundary_sample_count_after_dedup
    )

    per_candidate_stats = {
        "candidate_id": candidate_id,
        "support_count": len(exact_support_paragraph_ids),
        "boundary_sample_count_before_dedup": boundary_sample_count_before_dedup,
        "boundary_sample_count_after_dedup": boundary_sample_count_after_dedup,
        "boundary_overlap_count": boundary_overlap_count,
        "final_candidate_paragraph_count": len(final_candidate_paragraph_ids),
        "boundary_pool_sizes": boundary_pool_sizes,
        "sampled_boundary_sizes": sampled_boundary_sizes,
        "explore_field_missing_support_overlap_count": explore_field_missing_support_overlap_count,
        "support_overlap_count_by_error": support_overlap_count_by_error,
        "has_any_boundary": any(v > 0 for v in boundary_pool_sizes.values()),
    }

    return manifest_row, candidate_id, final_candidate_paragraph_ids, per_candidate_stats


def process_chunk(chunk, sample_size_per_error, base_seed):
    out = []
    for c in chunk:
        out.append(analyze_candidate_and_sample(
            candidate=c,
            sample_size_per_error=sample_size_per_error,
            base_seed=base_seed,
        ))
    return out


# =========================
# Stats builder
# =========================
def build_stats(manifest_rows, candidate_paragraph_ids_map, per_candidate_stats, sample_size_per_error, base_seed):
    error_types = [
        "fixed_constraint_missing_1",
        "fixed_constraint_mismatch_1",
        "negated_field_mismatch",
        "negated_field_missing",
        "explore_field_missing",
    ]

    stats = {
        "total_candidates": len(manifest_rows),
        "sample_size_per_error": sample_size_per_error,
        "base_seed": base_seed,
        "error_types": error_types,
    }

    support_counts = [x["support_count"] for x in per_candidate_stats]
    boundary_counts_before_dedup = [x["boundary_sample_count_before_dedup"] for x in per_candidate_stats]
    boundary_counts_after_dedup = [x["boundary_sample_count_after_dedup"] for x in per_candidate_stats]
    boundary_overlap_counts = [x["boundary_overlap_count"] for x in per_candidate_stats]
    final_candidate_counts = [x["final_candidate_paragraph_count"] for x in per_candidate_stats]
    explore_field_missing_support_overlap_counts = [
        x["explore_field_missing_support_overlap_count"] for x in per_candidate_stats
    ]

    stats["support_count_summary"] = summary_stats(support_counts)
    stats["boundary_sample_count_before_dedup_summary"] = summary_stats(boundary_counts_before_dedup)
    stats["boundary_sample_count_after_dedup_summary"] = summary_stats(boundary_counts_after_dedup)
    stats["boundary_overlap_count_summary"] = summary_stats(boundary_overlap_counts)
    stats["final_candidate_paragraph_count_summary"] = summary_stats(final_candidate_counts)
    stats["explore_field_missing_support_overlap_count_summary"] = summary_stats(
        explore_field_missing_support_overlap_counts
    )
    stats["explore_field_missing_support_overlap_sum"] = int(sum(explore_field_missing_support_overlap_counts))

    type_counts = {}
    for row in manifest_rows:
        t = row.get("type") or "UNKNOWN"
        type_counts[t] = type_counts.get(t, 0) + 1
    stats["type_counts"] = dict(sorted(type_counts.items()))

    pool_summary = {}
    sampled_summary = {}
    nonzero_pool_counts = {}
    nonzero_sampled_counts = {}

    for err in error_types:
        pool_values = [x["boundary_pool_sizes"][err] for x in per_candidate_stats]
        sampled_values = [x["sampled_boundary_sizes"][err] for x in per_candidate_stats]

        pool_summary[err] = {
            **summary_stats(pool_values),
            "sum": int(sum(pool_values)),
        }
        sampled_summary[err] = {
            **summary_stats(sampled_values),
            "sum": int(sum(sampled_values)),
        }
        nonzero_pool_counts[err] = sum(1 for v in pool_values if v > 0)
        nonzero_sampled_counts[err] = sum(1 for v in sampled_values if v > 0)

    stats["boundary_pool_size_summary_by_error"] = pool_summary
    stats["sampled_boundary_size_summary_by_error"] = sampled_summary
    stats["nonzero_pool_candidate_counts_by_error"] = nonzero_pool_counts
    stats["nonzero_sampled_candidate_counts_by_error"] = nonzero_sampled_counts

    # 放在这里再算总和
    support_count_sum = int(sum(support_counts))
    sampled_boundary_pair_sum = int(sum(sampled_summary[err]["sum"] for err in error_types))
    global_boundary_pair_sum = int(sum(pool_summary[err]["sum"] for err in error_types))

    sampled_total_query_paragraph_pairs = support_count_sum + sampled_boundary_pair_sum
    global_total_query_paragraph_pairs = support_count_sum + global_boundary_pair_sum

    stats["support_count_sum"] = support_count_sum
    stats["sampled_boundary_pair_sum"] = sampled_boundary_pair_sum
    stats["global_boundary_pair_sum"] = global_boundary_pair_sum
    stats["sampled_total_query_paragraph_pairs"] = sampled_total_query_paragraph_pairs
    stats["global_total_query_paragraph_pairs"] = global_total_query_paragraph_pairs

    dedup_lengths = [len(v) for v in candidate_paragraph_ids_map.values()]
    stats["dedup_candidate_paragraph_ids_summary"] = summary_stats(dedup_lengths)
    stats["total_dedup_candidate_paragraph_ids_sum"] = int(sum(dedup_lengths))

    stats["candidates_with_any_boundary_count"] = sum(
        1 for x in per_candidate_stats if x["has_any_boundary"]
    )
    stats["candidates_with_any_boundary_ratio"] = round(
        stats["candidates_with_any_boundary_count"] / len(per_candidate_stats), 4
    ) if per_candidate_stats else None

    type_level = {}
    grouped = {}
    for x in per_candidate_stats:
        cid = x["candidate_id"]
        row = next(r for r in manifest_rows if r["candidate_id"] == cid)
        t = row.get("type") or "UNKNOWN"
        grouped.setdefault(t, []).append(x)

    for t, rows in sorted(grouped.items()):
        item = {
            "candidate_count": len(rows),
            "support_count_summary": summary_stats([r["support_count"] for r in rows]),
            "boundary_sample_count_before_dedup_summary": summary_stats([r["boundary_sample_count_before_dedup"] for r in rows]),
            "boundary_sample_count_after_dedup_summary": summary_stats([r["boundary_sample_count_after_dedup"] for r in rows]),
            "boundary_overlap_count_summary": summary_stats([r["boundary_overlap_count"] for r in rows]),
            "final_candidate_paragraph_count_summary": summary_stats([r["final_candidate_paragraph_count"] for r in rows]),
            "explore_field_missing_support_overlap_count_summary": summary_stats(
                [r["explore_field_missing_support_overlap_count"] for r in rows]
            ),
            "boundary_pool_size_summary_by_error": {},
            "sampled_boundary_size_summary_by_error": {},
        }
        for err in error_types:
            pool_values = [r["boundary_pool_sizes"][err] for r in rows]
            sampled_values = [r["sampled_boundary_sizes"][err] for r in rows]
            item["boundary_pool_size_summary_by_error"][err] = {
                **summary_stats(pool_values),
                "sum": int(sum(pool_values)),
            }
            item["sampled_boundary_size_summary_by_error"][err] = {
                **summary_stats(sampled_values),
                "sum": int(sum(sampled_values)),
            }
        type_level[t] = item

    stats["type_level_summary"] = type_level
    return stats

# =========================
# Main
# =========================
def main(
    setup_records_path,
    sampled_constraints_path,
    queries_raw_path,
    boundary_manifest_output_path,
    candidate_paragraph_ids_output_path,
    stats_output_path,
    sample_size_per_error=20,
    base_seed=42,
    max_workers=None,
    chunk_size=50,
):
    candidates = list(read_jsonl(sampled_constraints_path))
    if not candidates:
        print(f"[WARN] No candidates found: {sampled_constraints_path}")
        return
    queries_raw = read_json(queries_raw_path)
    candidates = [c for c in candidates if c.get("candidate_id") in queries_raw]

    max_workers = max_workers or max(1, mp.cpu_count() - 1)
    chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]

    manifest_rows = []
    candidate_paragraph_ids_map = {}
    per_candidate_stats = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(setup_records_path,),
    ) as ex:
        futures = [
            ex.submit(process_chunk, chunk, sample_size_per_error, base_seed)
            for chunk in chunks
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Sampling boundary paragraphs"):
            results = fut.result()
            for manifest_row, candidate_id, pid_list, candidate_stat in results:
                manifest_rows.append(manifest_row)
                candidate_paragraph_ids_map[candidate_id] = pid_list
                per_candidate_stats.append(candidate_stat)

    manifest_rows.sort(key=lambda x: str(x.get("candidate_id", "")))
    candidate_paragraph_ids_map = {
        k: candidate_paragraph_ids_map[k]
        for k in sorted(candidate_paragraph_ids_map.keys())
    }
    per_candidate_stats.sort(key=lambda x: str(x.get("candidate_id", "")))

    stats = build_stats(
        manifest_rows=manifest_rows,
        candidate_paragraph_ids_map=candidate_paragraph_ids_map,
        per_candidate_stats=per_candidate_stats,
        sample_size_per_error=sample_size_per_error,
        base_seed=base_seed,
    )

    write_jsonl(boundary_manifest_output_path, manifest_rows)
    write_json(candidate_paragraph_ids_output_path, candidate_paragraph_ids_map)
    write_json(stats_output_path, stats)

    print(f"[STATS] support_count_sum = {stats['support_count_sum']}")
    print(f"[STATS] sampled_boundary_pair_sum = {stats['sampled_boundary_pair_sum']}")
    print(f"[STATS] global_boundary_pair_sum = {stats['global_boundary_pair_sum']}")
    print(f"[STATS] sampled_total_query_paragraph_pairs = {stats['sampled_total_query_paragraph_pairs']}")
    print(f"[STATS] global_total_query_paragraph_pairs = {stats['global_total_query_paragraph_pairs']}")

    print(f"[OK] Saved boundary manifest: {boundary_manifest_output_path}")
    print(f"[OK] Saved candidate paragraph ids: {candidate_paragraph_ids_output_path}")
    print(f"[OK] Saved stats: {stats_output_path}")
    print(f"[OK] Total processed sampled candidates: {len(manifest_rows)}")


if __name__ == "__main__":
    domains = [
        "electrochemistry",
        "cell_culture",
        "immunotechnology",
        "inorganic_materials",
        "organic_synthesis",
        "polymer_materials",
    ]

    # =========================
    # Hyperparameters
    # =========================
    sample_size_per_error = 5
    base_seed = 42

    max_workers = min(32, os.cpu_count() or 1, 64)
    chunk_size = 50

    for d in domains:
        print(f"\n=== Boundary sampling stage: {d} ===")

        setup_records_path = f"seed_data/{d}/constraints/setup_records.jsonl"
        sampled_constraints_path = f"seed_data/{d}/constraints/sampled_constraints_stage1.jsonl"
        queries_raw_path = f"data/{d}/gen_query/queries_raw.json"

        boundary_manifest_output_path = f"data/{d}/stage1_relevance/sampled_constraints_stage1.boundary_manifest.jsonl"
        candidate_paragraph_ids_output_path = f"data/{d}/stage1_relevance/sampled_constraints_stage1.candidate_paragraph_ids.json"
        stats_output_path = f"data/{d}/stage1_relevance/sampled_constraints_stage1.boundary_stats.json"

        main(
            setup_records_path=setup_records_path,
            sampled_constraints_path=sampled_constraints_path,
            queries_raw_path=queries_raw_path,
            boundary_manifest_output_path=boundary_manifest_output_path,
            candidate_paragraph_ids_output_path=candidate_paragraph_ids_output_path,
            stats_output_path=stats_output_path,
            sample_size_per_error=sample_size_per_error,
            base_seed=base_seed,
            max_workers=max_workers,
            chunk_size=chunk_size,
        )


# python scripts/stage1_relevance/get_boundary_paragraph.py
        

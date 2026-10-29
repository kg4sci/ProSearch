import json
import random
from pathlib import Path
from collections import defaultdict, Counter


DOMAINS = [
    "electrochemistry",
    "cell_culture",
    "immunotechnology",
    "inorganic_materials",
    "organic_synthesis",
    "polymer_materials",
]

PRIMARY_DOMAIN = "electrochemistry"

DOMAIN_QUOTAS = {
    d: (45 if d == PRIMARY_DOMAIN else 15)
    for d in DOMAINS
}

# 5:3:2 = exact_support : boundary_negative : non_boundary_negative
LABEL_WEIGHTS = {
    "exact_support": 5,
    "boundary_negative": 3,
    "non_boundary_negative": 2,
}

# 余数优先给 boundary_negative，再给 exact_support
# 所以：
# 45 -> 22 / 14 / 9
# 15 -> 7 / 5 / 3
REMAINDER_PRIORITY = [
    "boundary_negative",
    "exact_support",
    "non_boundary_negative",
]

INPUT_TEMPLATE = "data/{d}/stage1_relevance/paragraph_labels_from_judge.jsonl"
OUTPUT_SAMPLE_PATH = "data/audit_sample_from_paragraph_labels.jsonl"
OUTPUT_STATS_PATH = "data/audit_sample_from_paragraph_labels.stats.json"

RANDOM_SEED = 42

# 每个 candidate 最多抽 1 条
MAX_PER_CANDIDATE = 1


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


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def allocate_counts(total, weights, remainder_priority):
    total_weight = sum(weights.values())
    raw = {k: total * v / total_weight for k, v in weights.items()}
    counts = {k: int(raw[k]) for k in raw}

    remain = total - sum(counts.values())
    if remain <= 0:
        return counts

    by_frac = defaultdict(list)
    for k in raw:
        frac = raw[k] - int(raw[k])
        by_frac[frac].append(k)

    ordered_keys = []
    for frac in sorted(by_frac.keys(), reverse=True):
        group = by_frac[frac]
        group.sort(key=lambda k: remainder_priority.index(k) if k in remainder_priority else 999)
        ordered_keys.extend(group)

    idx = 0
    while remain > 0 and ordered_keys:
        counts[ordered_keys[idx % len(ordered_keys)]] += 1
        idx += 1
        remain -= 1

    return counts


def load_domain_rows(domain, path):
    rows = []
    for row in read_jsonl(path):
        label = row.get("paragraph_label")
        if label not in {"exact_support", "boundary_negative", "non_boundary_negative"}:
            continue

        rows.append({
            "domain": domain,
            "candidate_id": row["candidate_id"],
            "paragraph_id": str(row["paragraph_id"]),
            "type": row.get("type"),
            "type_id": row.get("type_id"),
            "task_family": row.get("task_family"),
            "paragraph_label": label,
            "boundary_types": sorted(set(row.get("boundary_types", []) or [])),
            "judge_paragraph_verdict": row.get("judge_paragraph_verdict"),
            "setup_labels": row.get("setup_labels", []),
        })
    return rows


def sample_one_label(rows, target_n, rng, used_candidate_counter):
    """
    对 exact_support / non_boundary_negative 这样的单层直接抽样。
    优先保证 candidate 多样性：每个 candidate 最多 1 条。
    """
    candidate_to_rows = defaultdict(list)
    for row in rows:
        candidate_to_rows[row["candidate_id"]].append(row)

    candidate_ids = list(candidate_to_rows.keys())
    rng.shuffle(candidate_ids)

    sampled = []
    for cid in candidate_ids:
        if len(sampled) >= target_n:
            break
        if used_candidate_counter[cid] >= MAX_PER_CANDIDATE:
            continue
        chosen = rng.choice(candidate_to_rows[cid])
        sampled.append(chosen)
        used_candidate_counter[cid] += 1

    return sampled


def sample_boundary_rows(rows, target_n, rng, used_candidate_counter):
    """
    boundary_negative:
    - 先尽量覆盖不同 boundary_types
    - 再回填
    """
    sampled = []
    selected_pairs = set()

    subtype_to_rows = defaultdict(list)
    for row in rows:
        for bt in row.get("boundary_types", []):
            subtype_to_rows[bt].append(row)

    subtype_order = sorted(
        subtype_to_rows.keys(),
        key=lambda t: len({r["candidate_id"] for r in subtype_to_rows[t]})
    )

    # 第一轮：按 boundary 子类轮转覆盖
    progress = True
    while len(sampled) < target_n and progress:
        progress = False
        for subtype in subtype_order:
            if len(sampled) >= target_n:
                break

            pool = subtype_to_rows[subtype][:]
            rng.shuffle(pool)

            picked = None
            for row in pool:
                cid = row["candidate_id"]
                pair = (cid, row["paragraph_id"])
                if pair in selected_pairs:
                    continue
                if used_candidate_counter[cid] >= MAX_PER_CANDIDATE:
                    continue
                picked = row
                break

            if picked is None:
                continue

            sampled.append({
                **picked,
                "boundary_focus_type": subtype,
            })
            selected_pairs.add((picked["candidate_id"], picked["paragraph_id"]))
            used_candidate_counter[picked["candidate_id"]] += 1
            progress = True

    # 第二轮：剩余 boundary pool 回填
    if len(sampled) < target_n:
        pool = rows[:]
        rng.shuffle(pool)

        for row in pool:
            if len(sampled) >= target_n:
                break

            cid = row["candidate_id"]
            pair = (cid, row["paragraph_id"])
            if pair in selected_pairs:
                continue
            if used_candidate_counter[cid] >= MAX_PER_CANDIDATE:
                continue

            focus_type = rng.choice(row["boundary_types"]) if row["boundary_types"] else None
            sampled.append({
                **row,
                "boundary_focus_type": focus_type,
            })
            selected_pairs.add(pair)
            used_candidate_counter[cid] += 1

    return sampled


def sample_domain(domain, input_path, quota, rng):
    rows = load_domain_rows(domain, input_path)

    used_candidate_counter = Counter()
    target_counts = allocate_counts(quota, LABEL_WEIGHTS, REMAINDER_PRIORITY)

    exact_rows = [r for r in rows if r["paragraph_label"] == "exact_support"]
    boundary_rows = [r for r in rows if r["paragraph_label"] == "boundary_negative"]
    non_boundary_rows = [r for r in rows if r["paragraph_label"] == "non_boundary_negative"]

    sampled = []

    sampled_exact = sample_one_label(
        rows=exact_rows,
        target_n=target_counts["exact_support"],
        rng=rng,
        used_candidate_counter=used_candidate_counter,
    )
    for r in sampled_exact:
        r["sample_label"] = "exact_support"
        r["boundary_focus_type"] = None
    sampled.extend(sampled_exact)

    sampled_boundary = sample_boundary_rows(
        rows=boundary_rows,
        target_n=target_counts["boundary_negative"],
        rng=rng,
        used_candidate_counter=used_candidate_counter,
    )
    for r in sampled_boundary:
        r["sample_label"] = "boundary_negative"
    sampled.extend(sampled_boundary)

    sampled_non_boundary = sample_one_label(
        rows=non_boundary_rows,
        target_n=target_counts["non_boundary_negative"],
        rng=rng,
        used_candidate_counter=used_candidate_counter,
    )
    for r in sampled_non_boundary:
        r["sample_label"] = "non_boundary_negative"
        r["boundary_focus_type"] = None
    sampled.extend(sampled_non_boundary)

    # 如果某一层不够，尝试全局回填
    if len(sampled) < quota:
        seen_pairs = {(r["candidate_id"], r["paragraph_id"]) for r in sampled}
        fallback_pool = rows[:]
        rng.shuffle(fallback_pool)

        for row in fallback_pool:
            if len(sampled) >= quota:
                break

            cid = row["candidate_id"]
            pair = (cid, row["paragraph_id"])
            if pair in seen_pairs:
                continue
            if used_candidate_counter[cid] >= MAX_PER_CANDIDATE:
                continue

            extra = dict(row)
            extra["sample_label"] = row["paragraph_label"]
            extra["boundary_focus_type"] = (
                rng.choice(row["boundary_types"]) if row["paragraph_label"] == "boundary_negative" and row["boundary_types"] else None
            )

            sampled.append(extra)
            seen_pairs.add(pair)
            used_candidate_counter[cid] += 1

    label_counts = Counter(r["sample_label"] for r in sampled)
    boundary_focus_counts = Counter(
        r["boundary_focus_type"]
        for r in sampled
        if r["sample_label"] == "boundary_negative" and r.get("boundary_focus_type") is not None
    )

    stats = {
        "domain": domain,
        "quota": quota,
        "target_counts": target_counts,
        "actual_total": len(sampled),
        "actual_label_counts": dict(label_counts),
        "actual_boundary_focus_counts": dict(boundary_focus_counts),
        "available_counts": {
            "exact_support": len(exact_rows),
            "boundary_negative": len(boundary_rows),
            "non_boundary_negative": len(non_boundary_rows),
        },
        "unique_candidate_count": len({r["candidate_id"] for r in sampled}),
        "max_per_candidate": MAX_PER_CANDIDATE,
    }

    return sampled, stats


def main():
    rng = random.Random(RANDOM_SEED)

    all_sampled = []
    all_stats = {
        "seed": RANDOM_SEED,
        "primary_domain": PRIMARY_DOMAIN,
        "domain_quotas": DOMAIN_QUOTAS,
        "label_weights": LABEL_WEIGHTS,
        "remainder_priority": REMAINDER_PRIORITY,
        "max_per_candidate": MAX_PER_CANDIDATE,
        "domains": {},
    }

    for d in DOMAINS:
        input_path = INPUT_TEMPLATE.format(d=d)
        if not Path(input_path).exists():
            print(f"[WARN] missing file: {input_path}")
            continue

        quota = DOMAIN_QUOTAS[d]
        sampled_rows, stats = sample_domain(
            domain=d,
            input_path=input_path,
            quota=quota,
            rng=rng,
        )

        all_sampled.extend(sampled_rows)
        all_stats["domains"][d] = stats

    all_sampled.sort(key=lambda x: (
        x["domain"],
        x["sample_label"],
        x["candidate_id"],
        x["paragraph_id"],
    ))

    global_label_counts = Counter(r["sample_label"] for r in all_sampled)
    global_boundary_focus_counts = Counter(
        r["boundary_focus_type"]
        for r in all_sampled
        if r["sample_label"] == "boundary_negative" and r.get("boundary_focus_type") is not None
    )

    all_stats["global"] = {
        "sample_total": len(all_sampled),
        "label_counts": dict(global_label_counts),
        "boundary_focus_counts": dict(global_boundary_focus_counts),
        "unique_candidate_count": len({(r["domain"], r["candidate_id"]) for r in all_sampled}),
    }

    write_jsonl(OUTPUT_SAMPLE_PATH, all_sampled)
    write_json(OUTPUT_STATS_PATH, all_stats)

    print(f"[OK] saved sample: {OUTPUT_SAMPLE_PATH}")
    print(f"[OK] saved stats: {OUTPUT_STATS_PATH}")
    print(f"[STATS] total sampled: {len(all_sampled)}")
    print(f"[STATS] label counts: {dict(global_label_counts)}")


if __name__ == "__main__":
    main()

# python scripts/stage1_relevance/sample_from_paragraph_labels.py
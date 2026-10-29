import json
from collections import defaultdict, Counter


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_support_map(boundary_data):
    """
    (candidate_id, paragraph_id) -> True
    """
    support = set()

    for row in boundary_data:
        cid = row["candidate_id"]
        pids = row.get("support_paragraph_ids", [])

        for pid in pids:
            support.add((cid, pid))

    return support

def build_boundary_map(boundary_data):
    """
    (candidate_id, paragraph_id) -> [boundary_types]
    """
    mapping = defaultdict(list)

    for row in boundary_data:
        cid = row["candidate_id"]
        pools = row.get("sampled_boundary_paragraph_ids", {})

        for btype, pids in pools.items():
            for pid in pids:
                mapping[(cid, pid)].append(btype)

    return mapping


def init_stat():
    return {
        "total": 0,
        "counts": Counter()
    }


def update_stat(stat, verdict):
    stat["total"] += 1
    stat["counts"][verdict] += 1


def finalize_stat(stat):
    total = stat["total"]
    ratio = {}
    for k, v in stat["counts"].items():
        ratio[k] = v / total if total > 0 else 0
    stat["ratio"] = ratio
    return stat

def compute_support_stats(judge_data, support_map):
    stats = {
        "overall": init_stat(),
        "reproduction": init_stat(),
        "exploration": init_stat(),
    }

    for row in judge_data:
        cid = row["candidate_id"]
        pid = row["paragraph_id"]
        qtype = row.get("task_family")

        verdict = row.get("parsed_output", {}).get("paragraph_verdict")
        if verdict is None:
            continue

        if (cid, pid) not in support_map:
            continue

        # overall
        update_stat(stats["overall"], verdict)

        # 分类型
        if qtype == "reproduction":
            update_stat(stats["reproduction"], verdict)

        elif qtype == "exploration":
            update_stat(stats["exploration"], verdict)

    # finalize
    for k in stats:
        stats[k] = finalize_stat(stats[k])

        # 加 accuracy
        total = stats[k]["total"]
        satisfy = stats[k]["counts"].get("satisfy", 0)

        stats[k]["accuracy"] = satisfy / total if total > 0 else 0

    return stats

def compute_stats(judge_data, boundary_map):
    """
    返回结构：
    {
      overall: {boundary_type -> stat}
      reproduction: {boundary_type -> stat}
      condition_exploration: {boundary_type -> stat}
    }
    """
    stats = {
        "overall": defaultdict(init_stat),
        "reproduction": defaultdict(init_stat),
        "exploration": defaultdict(init_stat),
    }

    for row in judge_data:
        cid = row["candidate_id"]
        pid = row["paragraph_id"]
        qtype = row.get("task_family")
        # print(cid)
        verdict = row.get("parsed_output", {}).get("paragraph_verdict")
        if verdict is None:
            continue

        btypes = boundary_map.get((cid, pid), [])
        if not btypes:
            continue

        for b in btypes:
            # overall
            update_stat(stats["overall"][b], verdict)

            # 分类型
            if qtype == "reproduction":
                update_stat(stats["reproduction"][b], verdict)

            elif qtype == "exploration":
                update_stat(stats["exploration"][b], verdict)

    # finalize ratio
    for group in stats:
        for b in stats[group]:
            stats[group][b] = finalize_stat(stats[group][b])

    return stats


if __name__ == "__main__":

    domains = [
        "electrochemistry",
        "cell_culture",
        "immunotechnology",
        "inorganic_materials",
        "organic_synthesis",
        "polymer_materials",
    ]

    all_results = {}

    for d in domains:

        judge_path = f"data/{d}/stage1_relevance/judge_results_stage1_gpt-5.4.jsonl"
        boundary_path = f"data/{d}/stage1_relevance/sampled_constraints_stage1.boundary_manifest.jsonl"

        judge_data = load_jsonl(judge_path)
        boundary_data = load_jsonl(boundary_path)

        boundary_map = build_boundary_map(boundary_data)

        stats = compute_stats(judge_data, boundary_map)

        support_map = build_support_map(boundary_data)

        support_stats = compute_support_stats(judge_data, support_map)

        all_results[d] = {
            "boundary": stats,
            "support": support_stats
        }

        print(f"Finished domain: {d}")

    # 保存
    output_path = "data/sample_stage1_boundary_paragraph_verdict_stats.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved to {output_path}")

# python scripts/stage1_relevance/get_boundary_paragraph_verdict_stats.py


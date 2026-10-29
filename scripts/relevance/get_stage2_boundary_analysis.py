import json
from pathlib import Path
from statistics import mean
from collections import Counter


BOUNDARY_ERROR_TYPES = [
    "logic_near_miss_missing_1",
    "logic_near_miss_mismatch_1",
    "negated_field_mismatch",
    "negated_field_missing",
    "answer_missing",
]


def load_json(path: str):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_qrels(qrels: dict) -> dict:
    """
    统一成:
    {
      "query_id": {
        "paragraph_id": 1,
        ...
      }
    }
    """
    normalized = {}

    for query_id, doc_map in qrels.items():
        if not isinstance(doc_map, dict):
            continue

        normalized_docs = {}
        for paragraph_id, label in doc_map.items():
            try:
                if int(label) == 1:
                    normalized_docs[str(paragraph_id)] = 1
            except Exception:
                continue

        normalized[str(query_id)] = normalized_docs

    return normalized


def normalize_boundary_dict(boundary_dict: dict) -> dict:
    """
    统一成:
    {
      "query_id": {
        "logic_near_miss_missing_1": {"123", "456"},
        "logic_near_miss_mismatch_1": {...},
        ...
      }
    }

    注意:
    - 同一个 paragraph_id 可能同时属于多个 error type
    - 同一个 error type 内部会去重
    """
    normalized = {}

    for query_id, error_map in boundary_dict.items():
        if not isinstance(error_map, dict):
            continue

        qid = str(query_id)
        normalized[qid] = {}

        for error_type in BOUNDARY_ERROR_TYPES:
            values = error_map.get(error_type, [])
            if not isinstance(values, list):
                values = []

            normalized[qid][error_type] = {str(x) for x in values}

    return normalized


def sort_doc_map(doc_map: dict) -> dict:
    def key_fn(x):
        return (0, int(x)) if str(x).isdigit() else (1, str(x))
    return dict(sorted(doc_map.items(), key=lambda kv: key_fn(kv[0])))


def merge_qrels(qrels_stage1: dict, qrels_stage2: dict) -> dict:
    merged = {}
    all_query_ids = sorted(set(qrels_stage1.keys()) | set(qrels_stage2.keys()))

    for query_id in all_query_ids:
        merged_docs = {}

        for src in (qrels_stage1.get(query_id, {}), qrels_stage2.get(query_id, {})):
            for paragraph_id, label in src.items():
                if int(label) == 1:
                    merged_docs[str(paragraph_id)] = 1

        merged[query_id] = sort_doc_map(merged_docs)

    return merged


def filter_qrels_by_support_range(qrels: dict, min_positive: int = None, max_positive: int = None):
    kept = {}
    removed_query_ids = []

    for query_id, doc_map in qrels.items():
        support = len(doc_map)

        if min_positive is not None and support < min_positive:
            removed_query_ids.append(query_id)
            continue
        if max_positive is not None and support > max_positive:
            removed_query_ids.append(query_id)
            continue

        kept[query_id] = doc_map

    return kept, removed_query_ids


def filter_qrels_by_query_ids(qrels: dict, keep_query_ids) -> dict:
    keep_query_ids = set(keep_query_ids)
    return {qid: qrels.get(qid, {}) for qid in sorted(keep_query_ids) if qid in qrels}


def summarize_qrels(qrels: dict) -> dict:
    support_sizes = [len(doc_map) for doc_map in qrels.values()]
    query_count = len(support_sizes)
    query_count_with_positive = sum(1 for x in support_sizes if x > 0)
    query_count_without_positive = sum(1 for x in support_sizes if x == 0)
    total_positive_support = sum(support_sizes)

    return {
        "query_count": query_count,
        "query_count_with_positive": query_count_with_positive,
        "query_count_without_positive": query_count_without_positive,
        "total_positive_support": total_positive_support,
        "avg_positive_support_per_query": round(total_positive_support / query_count, 4) if query_count else 0.0,
        "avg_positive_support_per_nonzero_query": round(
            mean([x for x in support_sizes if x > 0]), 4
        ) if query_count_with_positive else 0.0,
        "positive_support_min": min(support_sizes) if support_sizes else None,
        "positive_support_max": max(support_sizes) if support_sizes else None,
    }


def compute_increment_stats(qrels_stage1: dict, qrels_stage2: dict, qrels_merged: dict) -> dict:
    all_query_ids = sorted(set(qrels_stage1.keys()) | set(qrels_stage2.keys()) | set(qrels_merged.keys()))

    stage2_new_positive_pairs = 0
    queries_with_stage2_new_positive = 0
    per_query_new_support = []

    for query_id in all_query_ids:
        s1 = set(qrels_stage1.get(query_id, {}).keys())
        s2 = set(qrels_stage2.get(query_id, {}).keys())

        new_docs = s2 - s1
        new_count = len(new_docs)

        stage2_new_positive_pairs += new_count
        per_query_new_support.append(new_count)

        if new_count > 0:
            queries_with_stage2_new_positive += 1

    return {
        "stage2_new_positive_pairs_over_stage1": stage2_new_positive_pairs,
        "queries_with_stage2_new_positive": queries_with_stage2_new_positive,
        "avg_stage2_new_positive_per_query": round(mean(per_query_new_support), 4) if per_query_new_support else 0.0,
        "avg_stage2_new_positive_per_nonzero_query": round(
            mean([x for x in per_query_new_support if x > 0]), 4
        ) if queries_with_stage2_new_positive else 0.0,
    }


def compute_boundary_overlap_stats(qrels_stage1: dict, qrels_stage2: dict, kept_query_ids, boundary_dict: dict):
    """
    在保留后的 query 上，统计 stage2 相对 stage1 新增的正样本，
    有多少落在 boundary dict 中，并给出分布。

    输出包含两种口径:
    1. unique 命中: 一个新增正样本(query, paragraph)只算一次
    2. by_error_type_non_exclusive: 如果一个 paragraph 同时属于多个 error type，会分别计入各类型
    """
    kept_query_ids = sorted(set(kept_query_ids))
    total_new_positive_pairs = 0
    unique_boundary_matched_pairs = 0
    queries_with_new_positive = 0
    queries_with_boundary_overlap = 0

    error_type_counter = Counter()
    overlap_pattern_counter = Counter()
    per_query_details = {}

    for query_id in kept_query_ids:
        s1 = set(qrels_stage1.get(query_id, {}).keys())
        s2 = set(qrels_stage2.get(query_id, {}).keys())
        new_docs = sorted(s2 - s1)

        if new_docs:
            queries_with_new_positive += 1

        boundary_map = boundary_dict.get(query_id, {})
        query_matched_count = 0
        matched_doc_details = []

        for paragraph_id in new_docs:
            total_new_positive_pairs += 1

            matched_types = []
            for error_type in BOUNDARY_ERROR_TYPES:
                if paragraph_id in boundary_map.get(error_type, set()):
                    matched_types.append(error_type)

            if matched_types:
                unique_boundary_matched_pairs += 1
                query_matched_count += 1

                for error_type in matched_types:
                    error_type_counter[error_type] += 1

                pattern_key = " + ".join(sorted(matched_types))
                overlap_pattern_counter[pattern_key] += 1

                matched_doc_details.append({
                    "paragraph_id": paragraph_id,
                    "matched_error_types": matched_types,
                })

        if query_matched_count > 0:
            queries_with_boundary_overlap += 1

        if new_docs:
            per_query_details[query_id] = {
                "stage2_new_positive_count": new_docs,
                "boundary_overlap_count": query_matched_count,
                "matched_docs": matched_doc_details,
            }

    def safe_ratio(num, den):
        return round(num / den, 4) if den else 0.0

    non_exclusive_ratio_over_total_new = {
        et: safe_ratio(error_type_counter.get(et, 0), total_new_positive_pairs)
        for et in BOUNDARY_ERROR_TYPES
    }

    non_exclusive_ratio_over_matched_unique = {
        et: safe_ratio(error_type_counter.get(et, 0), unique_boundary_matched_pairs)
        for et in BOUNDARY_ERROR_TYPES
    }

    return {
        "total_stage2_new_positive_pairs_on_kept_queries": total_new_positive_pairs,
        "unique_new_positive_pairs_matching_any_boundary": unique_boundary_matched_pairs,
        "unique_new_positive_pairs_not_in_boundary": total_new_positive_pairs - unique_boundary_matched_pairs,
        "unique_boundary_match_ratio_over_new_positive": safe_ratio(
            unique_boundary_matched_pairs, total_new_positive_pairs
        ),

        "queries_with_stage2_new_positive": queries_with_new_positive,
        "queries_with_boundary_overlap": queries_with_boundary_overlap,
        "queries_with_stage2_new_positive_but_no_boundary_overlap": (
            queries_with_new_positive - queries_with_boundary_overlap
        ),

        "by_error_type_non_exclusive_counts": {
            et: error_type_counter.get(et, 0) for et in BOUNDARY_ERROR_TYPES
        },
        "by_error_type_non_exclusive_ratio_over_total_new_positive": non_exclusive_ratio_over_total_new,
        "by_error_type_non_exclusive_ratio_over_unique_boundary_matches": non_exclusive_ratio_over_matched_unique,

        "overlap_pattern_counts": dict(sorted(overlap_pattern_counter.items())),
        "per_query_details": per_query_details,
    }


def main():
    domains = [
        # "cell_culture",
        # "electrochemistry",
        # "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        "polymer_materials",
    ]

    # merged 后保留范围
    min_positive = 2
    max_positive = 12

    for domain in domains:
        print(f"\n[Domain] {domain}")

        qrels_stage1_path = f"data/{domain}/stage1_relevance/qrels_stage1.json"
        qrels_stage2_path = f"data/{domain}/relevance/qrels_stage2.json"
        boundary_dict_path = f"data/{domain}/stage1_relevance/sampled_constraints_stage1.full_boundary_dict.json"

        output_qrels_path = f"data/{domain}/gen_query/qrels_llm.json"
        output_stats_path = f"data/{domain}/gen_query/qrels_llm_stats.json"
        output_boundary_overlap_path = f"data/{domain}/gen_query/qrels_llm_boundary_overlap.json"

        qrels_stage1 = normalize_qrels(load_json(qrels_stage1_path))
        qrels_stage2 = normalize_qrels(load_json(qrels_stage2_path))
        boundary_dict = normalize_boundary_dict(load_json(boundary_dict_path))

        # 先合并全部
        qrels_merged_all = merge_qrels(qrels_stage1, qrels_stage2)

        # 再根据 merged 后 support 范围过滤
        qrels_merged_kept, removed_query_ids = filter_qrels_by_support_range(
            qrels_merged_all,
            min_positive=min_positive,
            max_positive=max_positive,
        )
        kept_query_ids = sorted(qrels_merged_kept.keys())

        # stage1 / stage2 也只保留这些 query，便于统计一致
        qrels_stage1_kept = filter_qrels_by_query_ids(qrels_stage1, kept_query_ids)
        qrels_stage2_kept = filter_qrels_by_query_ids(qrels_stage2, kept_query_ids)

        boundary_overlap_stats = compute_boundary_overlap_stats(
            qrels_stage1_kept,
            qrels_stage2_kept,
            kept_query_ids,
            boundary_dict,
        )

        stats = {
            "input_files": {
                "qrels_stage1_path": qrels_stage1_path,
                "qrels_stage2_path": qrels_stage2_path,
                "boundary_dict_path": boundary_dict_path,
            },
            "filter_config": {
                "min_positive_after_merge": min_positive,
                "max_positive_after_merge": max_positive,
            },
            "filter_summary": {
                "total_queries_before_filter": len(qrels_merged_all),
                "kept_query_count": len(kept_query_ids),
                "removed_query_count": len(removed_query_ids),
                "removed_query_ids": removed_query_ids,
            },
            "stage1": summarize_qrels(qrels_stage1_kept),
            "stage2": summarize_qrels(qrels_stage2_kept),
            "merged": summarize_qrels(qrels_merged_kept),
            "increment": compute_increment_stats(qrels_stage1_kept, qrels_stage2_kept, qrels_merged_kept),
            "boundary_overlap_on_kept_queries": boundary_overlap_stats,
        }

        save_json(output_qrels_path, qrels_merged_kept)
        save_json(output_stats_path, stats)
        save_json(output_boundary_overlap_path, boundary_overlap_stats)

        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"[Saved] filtered merged qrels -> {output_qrels_path}")
        print(f"[Saved] filtered merged stats -> {output_stats_path}")
        print(f"[Saved] boundary overlap stats -> {output_boundary_overlap_path}")


if __name__ == "__main__":
    main()

    
# python scripts/relevance/get_stage2_boundary_analysis.py
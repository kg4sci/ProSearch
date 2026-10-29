import json
import argparse
from pathlib import Path
from statistics import mean
from collections import OrderedDict


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


def get_topk_doc_ids(doc_scores: dict, top_k: int):
    if not isinstance(doc_scores, dict):
        return []
    return list(doc_scores.keys())[:top_k]


def collect_all_query_ids(candidate_pool: dict, qrels: dict, runs_by_model: dict):
    query_ids = set(candidate_pool.keys()) | set(qrels.keys())
    for _, run in runs_by_model.items():
        query_ids.update(run.keys())
    return sorted(query_ids)


def safe_set_from_list(x):
    if isinstance(x, list):
        return set(str(v) for v in x)
    return set()


def safe_set_from_qrels_dict(x):
    if isinstance(x, dict):
        return set(str(k) for k, v in x.items() if v == 1)
    return set()


def ordered_union_topk_for_query(query_id: str, runs_by_model: dict, models: list[str], top_k: int):
    """
    多模型 union，保留顺序：
    按 models 的顺序依次读取各模型 top_k，
    同一个 doc 只保留第一次出现的位置。
    """
    seen = set()
    merged = []

    for model in models:
        run = runs_by_model.get(model, {})
        top_docs = get_topk_doc_ids(run.get(query_id, {}), top_k)
        for doc_id in top_docs:
            doc_id = str(doc_id)
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc_id)

    return merged


def summarize_count_list(values):
    if not values:
        return {
            "avg": 0.0,
            "min": 0,
            "max": 0,
            "sum": 0,
        }
    return {
        "avg": round(mean(values), 4),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def compute_model_stats_for_topk(
    query_ids: list[str],
    run: dict,
    candidate_pool: dict,
    qrels: dict,
    top_k: int,
):
    """
    针对单个模型、单个 top_k 的统计。
    """
    topk_in_stage1_candidate = []
    topk_in_stage1_positive = []
    topk_new_to_judge = []

    queries_with_any_stage1_positive_hit = 0
    queries_with_any_new_to_judge = 0

    for qid in query_ids:
        c1 = safe_set_from_list(candidate_pool.get(qid, []))
        p1 = safe_set_from_qrels_dict(qrels.get(qid, {}))
        top_docs = [str(x) for x in get_topk_doc_ids(run.get(qid, {}), top_k)]
        top_set = set(top_docs)

        in_candidate = len(top_set & c1)
        in_positive = len(top_set & p1)
        new_to_judge = len(top_set - c1)

        topk_in_stage1_candidate.append(in_candidate)
        topk_in_stage1_positive.append(in_positive)
        topk_new_to_judge.append(new_to_judge)

        if in_positive > 0:
            queries_with_any_stage1_positive_hit += 1
        if new_to_judge > 0:
            queries_with_any_new_to_judge += 1

    return {
        "n_queries": len(query_ids),
        "top_k": top_k,
        "avg_topk_in_stage1_candidate": round(mean(topk_in_stage1_candidate), 4) if query_ids else 0.0,
        "avg_topk_in_stage1_positive": round(mean(topk_in_stage1_positive), 4) if query_ids else 0.0,
        "avg_topk_new_to_judge": round(mean(topk_new_to_judge), 4) if query_ids else 0.0,
        "queries_with_any_stage1_positive_hit": queries_with_any_stage1_positive_hit,
        "queries_with_any_new_to_judge": queries_with_any_new_to_judge,
        "topk_in_stage1_candidate_summary": summarize_count_list(topk_in_stage1_candidate),
        "topk_in_stage1_positive_summary": summarize_count_list(topk_in_stage1_positive),
        "topk_new_to_judge_summary": summarize_count_list(topk_new_to_judge),
    }


def compute_union_outputs_and_stats(
    query_ids: list[str],
    runs_by_model: dict,
    models: list[str],
    candidate_pool: dict,
    qrels: dict,
    top_k: int,
):
    """
    计算：
    1) union_top{K}.json
    2) pooling_top{K}.json
    3) union 的统计
    """
    union_top = OrderedDict()
    pooling_top = OrderedDict()

    union_sizes = []
    union_in_stage1_candidate = []
    union_in_stage1_positive = []
    union_new_to_judge = []

    queries_with_any_stage1_positive_hit = 0
    queries_with_any_new_to_judge = 0

    all_unique_new_paragraph_ids = set()
    total_new_query_doc_pairs_to_judge = 0

    for qid in query_ids:
        c1 = safe_set_from_list(candidate_pool.get(qid, []))
        p1 = safe_set_from_qrels_dict(qrels.get(qid, {}))

        union_docs = ordered_union_topk_for_query(
            query_id=qid,
            runs_by_model=runs_by_model,
            models=models,
            top_k=top_k,
        )
        union_set = set(union_docs)

        new_docs = [doc_id for doc_id in union_docs if doc_id not in c1]

        union_top[qid] = union_docs
        pooling_top[qid] = new_docs

        in_candidate = len(union_set & c1)
        in_positive = len(union_set & p1)
        new_to_judge = len(new_docs)

        union_sizes.append(len(union_docs))
        union_in_stage1_candidate.append(in_candidate)
        union_in_stage1_positive.append(in_positive)
        union_new_to_judge.append(new_to_judge)

        if in_positive > 0:
            queries_with_any_stage1_positive_hit += 1
        if new_to_judge > 0:
            queries_with_any_new_to_judge += 1

        total_new_query_doc_pairs_to_judge += new_to_judge
        all_unique_new_paragraph_ids.update(new_docs)

    stats = {
        "n_queries": len(query_ids),
        "top_k": top_k,
        "avg_union_size": round(mean(union_sizes), 4) if query_ids else 0.0,
        "avg_union_in_stage1_candidate": round(mean(union_in_stage1_candidate), 4) if query_ids else 0.0,
        "avg_union_in_stage1_positive": round(mean(union_in_stage1_positive), 4) if query_ids else 0.0,
        "avg_union_new_to_judge": round(mean(union_new_to_judge), 4) if query_ids else 0.0,
        "queries_with_any_stage1_positive_hit": queries_with_any_stage1_positive_hit,
        "queries_with_any_new_to_judge": queries_with_any_new_to_judge,
        "total_new_query_doc_pairs_to_judge": total_new_query_doc_pairs_to_judge,
        "total_unique_new_paragraph_ids": len(all_unique_new_paragraph_ids),
        "union_size_summary": summarize_count_list(union_sizes),
        "union_in_stage1_candidate_summary": summarize_count_list(union_in_stage1_candidate),
        "union_in_stage1_positive_summary": summarize_count_list(union_in_stage1_positive),
        "union_new_to_judge_summary": summarize_count_list(union_new_to_judge),
    }

    return union_top, pooling_top, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="cell_culture")
    parser.add_argument("--models", type=str, default="bm25,nv_embed,bge_reasoner")
    parser.add_argument("--top_k_list", type=str, default="10,20,30,40,50")
    parser.add_argument("--save_top_k", type=int, default=50)
    args = parser.parse_args()

    domain = args.domain
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    top_k_list = [int(x.strip()) for x in args.top_k_list.split(",") if x.strip()]
    save_top_k = args.save_top_k

    if save_top_k not in top_k_list:
        top_k_list = sorted(set(top_k_list + [save_top_k]))

    candidate_pool_path = f"data/{domain}/stage1_relevance/sampled_constraints_stage1.candidate_paragraph_ids.json"
    qrels_path = f"data/{domain}/stage1_relevance/qrels_stage1.json"

    candidate_pool = load_json(candidate_pool_path)
    qrels = load_json(qrels_path)

    runs_by_model = {}
    for model in models:
        run_path = f"data/{domain}/eval/{model}/run_top100_raw.json"
        runs_by_model[model] = load_json(run_path)

    query_ids = collect_all_query_ids(candidate_pool, qrels, runs_by_model)

    print(f"[Domain] {domain}")
    print(f"[Models] {models}")
    print(f"[Queries] {len(query_ids)}")

    all_stats = {
        "domain": domain,
        "models": models,
        "n_queries": len(query_ids),
        "candidate_pool_path": candidate_pool_path,
        "qrels_path": qrels_path,
        "per_model": {},
        "union": {},
    }

    # 1) 每个模型、每个 top_k 的统计
    for model in models:
        all_stats["per_model"][model] = {}
        run = runs_by_model.get(model, {})

        for top_k in top_k_list:
            stats = compute_model_stats_for_topk(
                query_ids=query_ids,
                run=run,
                candidate_pool=candidate_pool,
                qrels=qrels,
                top_k=top_k,
            )
            all_stats["per_model"][model][f"top{top_k}"] = stats

            print(
                f"[Model={model}][top_k={top_k}] "
                f"avg_in_candidate={stats['avg_topk_in_stage1_candidate']:.2f}, "
                f"avg_in_positive={stats['avg_topk_in_stage1_positive']:.2f}, "
                f"avg_new_to_judge={stats['avg_topk_new_to_judge']:.2f}"
            )

    # 2) union 统计；并在 save_top_k 时保存两个池文件
    union_output = None
    pooling_output = None

    for top_k in top_k_list:
        union_top, pooling_top, union_stats = compute_union_outputs_and_stats(
            query_ids=query_ids,
            runs_by_model=runs_by_model,
            models=models,
            candidate_pool=candidate_pool,
            qrels=qrels,
            top_k=top_k,
        )
        all_stats["union"][f"top{top_k}"] = union_stats

        print(
            f"[Union][top_k={top_k}] "
            f"avg_union_size={union_stats['avg_union_size']:.2f}, "
            f"avg_in_candidate={union_stats['avg_union_in_stage1_candidate']:.2f}, "
            f"avg_in_positive={union_stats['avg_union_in_stage1_positive']:.2f}, "
            f"avg_new_to_judge={union_stats['avg_union_new_to_judge']:.2f}, "
            f"total_new_pairs={union_stats['total_new_query_doc_pairs_to_judge']}, "
            f"total_unique_new_para={union_stats['total_unique_new_paragraph_ids']}"
        )

        if top_k == save_top_k:
            union_output = union_top
            pooling_output = pooling_top

    # 3) 保存输出文件
    stats_path = f"data/{domain}/relevance/retrieval_overlap_stats.json"
    union_path = f"data/{domain}/relevance/union_top{save_top_k}.json"
    pooling_path = f"data/{domain}/relevance/pooling_top{save_top_k}.json"

    save_json(stats_path, all_stats)
    save_json(union_path, union_output if union_output is not None else {})
    save_json(pooling_path, pooling_output if pooling_output is not None else {})

    print(f"[Saved] Stats -> {stats_path}")
    print(f"[Saved] Union -> {union_path}")
    print(f"[Saved] Pooling -> {pooling_path}")


if __name__ == "__main__":
    main()



# cd /home/liangwenliang/检索数据集/
# python scripts/relevance/get_pooling_raw.py --domain "polymer_materials"
    
    
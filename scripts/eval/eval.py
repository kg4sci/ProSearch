import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import pytrec_eval


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_run(run: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for qid, doc_score_map in run.items():
        out[str(qid)] = {str(doc_id): float(score) for doc_id, score in doc_score_map.items()}
    return out


def normalize_qrels(qrels: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out = {}
    for qid, doc_rel_map in qrels.items():
        out[str(qid)] = {str(doc_id): int(rel) for doc_id, rel in doc_rel_map.items()}
    return out


def subset_dict_by_qids(d: Dict[str, Any], qids: List[str]) -> Dict[str, Any]:
    return {qid: d[qid] for qid in qids if qid in d}


def evaluate_run(qrels, run, metrics):
    # print("Requested metrics:", metrics)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))
    per_query_scores = evaluator.evaluate(run)

    if not per_query_scores:
        return pd.DataFrame(), {}

    per_query_df = pd.DataFrame(per_query_scores).T
    # print("Returned metric columns:", per_query_df.columns.tolist())

    mean_scores = per_query_df.mean(axis=0).to_dict()
    return per_query_df, mean_scores


def get_support_count_from_qrels(qrels_for_one_query: Dict[str, int]) -> int:
    return sum(1 for rel in qrels_for_one_query.values() if int(rel) > 0)


def support_to_bucket(support: int) -> str:
    if support == 0:
        return "0"
    elif 1 <= support <= 2:
        return "1-2"
    elif 3 <= support <= 10:
        return "3-10"
    elif 11 <= support <= 20:
        return "11-20"
    elif 21 <= support <= 30:
        return "21-30"
    elif 31 <= support <= 50:
        return "31-50"
    else:
        return "51+"


def enrich_queries_with_support(
    queries_json: Dict[str, Dict[str, Any]],
    qrels: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, Any]]:
    """
    为每个 query 补两个字段：
      - support_count
      - support_bucket
    """
    enriched = {}

    all_qids = set(queries_json.keys()) | set(qrels.keys())

    for qid in all_qids:
        meta = dict(queries_json.get(qid, {}))
        support_count = get_support_count_from_qrels(qrels.get(qid, {}))
        meta["support_count"] = support_count
        meta["support_bucket"] = support_to_bucket(support_count)
        enriched[str(qid)] = meta

    return enriched


def filter_qids_by_group(queries_json: Dict[str, Dict[str, Any]], field: str, value: Any):
    return [str(qid) for qid, meta in queries_json.items() if meta.get(field) == value]


def sort_group_values(field: str, values: List[Any]) -> List[Any]:
    """
    让 support_bucket 的输出顺序更自然，其它字段保持稳定排序。
    """
    if field == "support_bucket":
        bucket_order = {
            "0": 0,
            "1-2": 1,
            "3-10": 2,
            "11-20": 3,
            "21-30": 4,
            "31-50": 5,
            "51+": 6,
        }
        return sorted(values, key=lambda x: bucket_order.get(str(x), 999))

    def key_fn(x):
        if isinstance(x, bool):
            return (0, str(x))
        try:
            return (1, float(x))
        except Exception:
            return (2, str(x))

    return sorted(values, key=key_fn)


def evaluate_one(domain, model, queries_path, qrels_path, run_path, metrics, group_fields):
    queries_json = load_json(queries_path)
    qrels = normalize_qrels(load_json(qrels_path))
    run = normalize_run(load_json(run_path))

    valid_qids = sorted(set(qrels.keys()) & set(run.keys()))
    if not valid_qids:
        print(f"警告: domain={domain}, model={model} 没有重合的 QID")
        return pd.DataFrame()

    qrels = subset_dict_by_qids(qrels, valid_qids)
    run = subset_dict_by_qids(run, valid_qids)

    # 新增：根据 qrels 动态补 support_count / support_bucket
    queries_json = enrich_queries_with_support(queries_json, qrels)

    rows = []

    # 1. Global Metrics
    _, global_scores = evaluate_run(qrels, run, metrics)
    base_info = {"domain": domain, "model": model}
    rows.append({
        **base_info,
        "group_field": "__global__",
        "group_value": "__global__",
        "n_queries": len(valid_qids),
        **global_scores
    })

    # 2. Grouped Metrics
    for field in group_fields:
        unique_values = set(
            queries_json.get(qid, {}).get(field)
            for qid in valid_qids
            if field in queries_json.get(qid, {})
        )
        unique_values = sort_group_values(field, list(unique_values))

        for val in unique_values:
            target_qids = filter_qids_by_group(queries_json, field, val)
            intersect_qids = sorted(set(target_qids) & set(valid_qids))

            if not intersect_qids:
                continue

            sub_qrels = subset_dict_by_qids(qrels, intersect_qids)
            sub_run = subset_dict_by_qids(run, intersect_qids)
            _, mean_scores = evaluate_run(sub_qrels, sub_run, metrics)

            rows.append({
                **base_info,
                "group_field": field,
                "group_value": str(val),
                "n_queries": len(intersect_qids),
                **mean_scores
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", type=str, required=True)
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--group_fields", type=str, default="")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--queries_relpath", type=str, required=True)
    parser.add_argument("--qrels_relpath", type=str, required=True)
    parser.add_argument("--run_relpath", type=str, required=True)
    args = parser.parse_args()

    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    group_fields = [x.strip() for x in args.group_fields.split(",") if x.strip()]

    all_dfs = []

    for d in domains:
        for m in models:
            q_p = args.queries_relpath.replace("{domain}", d).replace("{model}", m)
            ql_p = args.qrels_relpath.replace("{domain}", d).replace("{model}", m)
            r_p = args.run_relpath.replace("{domain}", d).replace("{model}", m)

            if not (Path(q_p).exists() and Path(ql_p).exists() and Path(r_p).exists()):
                print(f"跳过路径缺失: {d} / {m}")
                continue

            print(f"正在计算: Domain={d}, Model={m}")
            df = evaluate_one(d, m, q_p, ql_p, r_p, metrics, group_fields)
            all_dfs.append(df)

    if not all_dfs:
        print("未生成任何有效评估数据。")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    # 精度控制：保留3位小数
    num_cols = final_df.select_dtypes(include=["float64", "float32"]).columns
    final_df[num_cols] = final_df[num_cols].round(3)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"成功保存结果至: {output_path}")


if __name__ == "__main__":
    main()
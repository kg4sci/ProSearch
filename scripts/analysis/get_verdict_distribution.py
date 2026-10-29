import os
import json
import csv
from collections import defaultdict

domain = "electrochemistry"
QRELS_PATH = f"data/{domain}/gen_query/qrels_llm_stats.json"   # 当前脚本里不强依赖，可保留
JUDGMENTS_PATH = f"data/{domain}/relevance/judgments_top50.jsonl"

MODEL_LIST = [
    "bm25",
    "bge_reasoner",
    "nv_embed",
]

K_LIST = [10, 20, 30, 40, 50]

OUTPUT_SUMMARY_JSON = f"data/{domain}/analysis/topk_query_macro_verdict_distribution.json"
OUTPUT_SUMMARY_CSV = f"data/{domain}/analysis/topk_query_macro_verdict_distribution.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_judgments(judgments_path):
    """
    构建:
        verdict_map[(query_id, paragraph_id)] = paragraph_verdict
    """
    verdict_map = {}
    total = 0
    success = 0

    with open(judgments_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)

            query_id = str(obj["query_id"])
            paragraph_id = str(obj["paragraph_id"])
            paragraph_verdict = obj.get("paragraph_verdict")

            if paragraph_verdict is None:
                continue

            verdict_map[(query_id, paragraph_id)] = paragraph_verdict
            success += 1

    print(f"[Info] Loaded judgments: total_lines={total}, success_pairs={success}")
    return verdict_map


def get_topk_items(run_dict_for_one_query, k):
    """
    run_top100_raw.json 示例:
    {
      "POS_000650": {
        "49970": 0.6745651960372925,
        "27821": 0.666388988494873
      }
    }

    为稳妥起见，按 score 降序重排后取前 k。
    """
    items = list(run_dict_for_one_query.items())
    items = sorted(items, key=lambda x: x[1], reverse=True)
    return items[:k]


def safe_div(x, y):
    return x / y if y > 0 else 0.0


def analyze_one_model_macro_by_query(model_name, run_path, verdict_map, k_list):
    """
    返回:
      summary_results[k] = 宏平均结果
      per_query_results[k] = 每个query的详细比例
    """
    run_data = load_json(run_path)
    query_ids = list(run_data.keys())

    summary_results = {}
    per_query_results = {}

    for k in k_list:
        query_level_rows = []

        for query_id in query_ids:
            para_score_dict = run_data[query_id]
            topk = get_topk_items(para_score_dict, k)

            counts = defaultdict(int)

            for paragraph_id, _score in topk:
                paragraph_id = str(paragraph_id)
                verdict = verdict_map.get((str(query_id), paragraph_id), None)

                if verdict == "satisfy":
                    counts["satisfy_count"] += 1
                elif verdict == "uncertain":
                    counts["uncertain_count"] += 1
                elif verdict == "violate":
                    counts["violate_count"] += 1
                else:
                    counts["missing_judgment_count"] += 1

            total = len(topk)

            row = {
                "model": model_name,
                "k": k,
                "query_id": query_id,
                "topk_size": total,
                "satisfy_count": counts["satisfy_count"],
                "uncertain_count": counts["uncertain_count"],
                "violate_count": counts["violate_count"],
                "missing_judgment_count": counts["missing_judgment_count"],
                "satisfy_ratio": safe_div(counts["satisfy_count"], total),
                "uncertain_ratio": safe_div(counts["uncertain_count"], total),
                "violate_ratio": safe_div(counts["violate_count"], total),
                "missing_judgment_ratio": safe_div(counts["missing_judgment_count"], total),
            }
            query_level_rows.append(row)

        # 对 query 的 ratio 做宏平均
        num_queries = len(query_level_rows)

        avg_satisfy_ratio = safe_div(
            sum(r["satisfy_ratio"] for r in query_level_rows), num_queries
        )
        avg_uncertain_ratio = safe_div(
            sum(r["uncertain_ratio"] for r in query_level_rows), num_queries
        )
        avg_violate_ratio = safe_div(
            sum(r["violate_ratio"] for r in query_level_rows), num_queries
        )
        avg_missing_judgment_ratio = safe_div(
            sum(r["missing_judgment_ratio"] for r in query_level_rows), num_queries
        )

        # 也顺手保留 query-level count 的平均值
        avg_satisfy_count = safe_div(
            sum(r["satisfy_count"] for r in query_level_rows), num_queries
        )
        avg_uncertain_count = safe_div(
            sum(r["uncertain_count"] for r in query_level_rows), num_queries
        )
        avg_violate_count = safe_div(
            sum(r["violate_count"] for r in query_level_rows), num_queries
        )
        avg_missing_judgment_count = safe_div(
            sum(r["missing_judgment_count"] for r in query_level_rows), num_queries
        )

        summary_results[k] = {
            "model": model_name,
            "k": k,
            "num_queries": num_queries,
            "avg_satisfy_count": avg_satisfy_count,
            "avg_uncertain_count": avg_uncertain_count,
            "avg_violate_count": avg_violate_count,
            "avg_missing_judgment_count": avg_missing_judgment_count,
            "avg_satisfy_ratio": avg_satisfy_ratio,
            "avg_uncertain_ratio": avg_uncertain_ratio,
            "avg_violate_ratio": avg_violate_ratio,
            "avg_missing_judgment_ratio": avg_missing_judgment_ratio,
        }

        per_query_results[k] = query_level_rows

    return summary_results, per_query_results


def save_summary_json(all_summary_results, output_json):
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_summary_results, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved summary JSON to: {output_json}")


def save_per_query_json(all_per_query_results, output_json):
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_per_query_results, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved per-query JSON to: {output_json}")


def save_summary_csv(all_summary_results, output_csv):
    fieldnames = [
        "model",
        "k",
        "num_queries",
        "avg_satisfy_count",
        "avg_uncertain_count",
        "avg_violate_count",
        "avg_missing_judgment_count",
        "avg_satisfy_ratio",
        "avg_uncertain_ratio",
        "avg_violate_ratio",
        "avg_missing_judgment_ratio",
    ]

    rows = []
    for model_name, res_by_k in all_summary_results.items():
        for k, row in res_by_k.items():
            rows.append(row)

    rows = sorted(rows, key=lambda x: (x["model"], x["k"]))

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Info] Saved summary CSV to: {output_csv}")


def pretty_print(all_summary_results):
    print("\n===== Query-macro Top-k Verdict Distribution =====")
    for model_name, res_by_k in all_summary_results.items():
        print(f"\n### Model: {model_name}")
        for k in sorted(res_by_k.keys()):
            r = res_by_k[k]
            print(
                f"k={k:<3d} | "
                f"avg_satisfy={r['avg_satisfy_count']:.4f} ({r['avg_satisfy_ratio']:.4f}) | "
                f"avg_uncertain={r['avg_uncertain_count']:.4f} ({r['avg_uncertain_ratio']:.4f}) | "
                f"avg_violate={r['avg_violate_count']:.4f} ({r['avg_violate_ratio']:.4f}) | "
                f"avg_missing={r['avg_missing_judgment_count']:.4f} ({r['avg_missing_judgment_ratio']:.4f})"
            )


def main():
    if os.path.exists(QRELS_PATH):
        _ = load_json(QRELS_PATH)
        print(f"[Info] Loaded qrels from: {QRELS_PATH}")
    else:
        print(f"[Warning] Qrels file not found: {QRELS_PATH} (not required for this analysis)")

    verdict_map = load_judgments(JUDGMENTS_PATH)

    all_summary_results = {}
    all_per_query_results = {}

    for model_name in MODEL_LIST:
        run_path = os.path.join('data', domain, 'eval', model_name, "run_top100_raw.json")
        if not os.path.exists(run_path):
            print(f"[Warning] Missing run file for model={model_name}: {run_path}")
            continue

        print(f"[Info] Processing model: {model_name}")
        summary_results, per_query_results = analyze_one_model_macro_by_query(
            model_name=model_name,
            run_path=run_path,
            verdict_map=verdict_map,
            k_list=K_LIST,
        )
        all_summary_results[model_name] = summary_results
        all_per_query_results[model_name] = per_query_results

    pretty_print(all_summary_results)
    save_summary_json(all_summary_results, OUTPUT_SUMMARY_JSON)
    save_summary_csv(all_summary_results, OUTPUT_SUMMARY_CSV)


if __name__ == "__main__":
    main()

# python scripts/analysis/get_verdict_distribution.py
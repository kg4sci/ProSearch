import json
import argparse
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


domain = "electrochemistry"
ks = [50, 40, 30, 20]
models = ["bm25", "nv_embed", "bge_reasoner"]

qrels_path = f"data/{domain}/gen_query/qrels_llm.json"
run_path_template = "data/{domain}/eval/{model}/run_top100_raw.json"
output_csv = f"result/tables/{domain}_pool_coverage_k.csv"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_topk_docs(run: Dict[str, Dict[str, float]], k: int) -> Dict[str, List[str]]:
    """
    run_top100_raw.json 已按分数降序排列。
    直接取前 k 个 doc_id。
    """
    out = {}
    for qid, doc_score_map in run.items():
        out[str(qid)] = [str(doc_id) for doc_id in list(doc_score_map.keys())[:k]]
    return out


def union_topk_docs(models: List[str], qid: str, model_runs_topk: Dict[str, Dict[str, List[str]]]) -> Set[str]:
    """
    对一个 query，把多个模型的 top-k 文档做 union。
    """
    doc_set = set()
    for model in models:
        doc_set.update(model_runs_topk[model].get(qid, []))
    
    return doc_set


def evaluate_pool_at_k(
    qrels: Dict[str, Set[str]],
    model_runs: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    k: int,
):
    # 每个模型先截到 top-k
    model_runs_topk = {
        model: get_topk_docs(model_runs[model], k)
        for model in models
    }

    valid_qids = set(qrels.keys())
    for model in models:
        valid_qids &= set(model_runs_topk[model].keys())
    valid_qids = sorted(valid_qids)

    total_positives = 0
    covered_positives = 0
    queries_with_hit = 0
    union_sizes = []

    for qid in valid_qids:
        pos_docs = qrels[qid]
        retrieved_docs = union_topk_docs(models, qid, model_runs_topk)
        union_sizes.append(len(retrieved_docs))
        hit_docs = pos_docs & retrieved_docs

        total_positives += len(pos_docs)
        covered_positives += len(hit_docs)

        if len(hit_docs) > 0:
            queries_with_hit += 1

    n_queries = len(valid_qids)
    coverage_at_k = covered_positives / total_positives if total_positives else 0.0
    query_hit_rate = queries_with_hit / n_queries if n_queries else 0.0

    return {
        "n_queries": n_queries,
        "union_topk_size": sum(union_sizes),
        "covered_positives_at_k": covered_positives,
        "coverage_at_k": round(coverage_at_k, 4),
        "queries_with_hit": queries_with_hit,
        "query_hit_rate": round(query_hit_rate, 4),
    }


def main():

    qrels_raw = load_json(qrels_path)
    qrels = {
        str(qid): {str(doc_id) for doc_id, rel in doc_rel_map.items() if int(rel) > 0}
        for qid, doc_rel_map in qrels_raw.items()
    }

    model_runs = {}
    for model in models:
        run_path = run_path_template.replace("{domain}", domain).replace("{model}", model)
        if not Path(run_path).exists():
            raise FileNotFoundError(f"Run file not found: {run_path}")
        model_runs[model] = load_json(run_path)

    results = {}
    for k in ks:
        results[f"k={k}"] = evaluate_pool_at_k(qrels, model_runs, models, k)
        
    df = pd.DataFrame(results)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).T
    df.to_csv(output_path)

    print(df.to_string())
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()


# python scripts/analysis/get_pool_coverage_k.py
    
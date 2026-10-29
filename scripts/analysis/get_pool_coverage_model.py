import json
import itertools
from pathlib import Path
import pandas as pd

# 不同模型贡献分析

DOMAIN = "electrochemistry"
TOP_K = 50
MODELS = ["bm25", "nv_embed", "bge_reasoner"]

QRELS_PATH = f"data/{DOMAIN}/gen_query/qrels_llm.json"
RUN_PATH_TEMPLATE = "data/{domain}/eval/{model}/run_top100_raw.json"
OUT_PATH = f"result/tables/{DOMAIN}_pool_coverage_model.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_topk_docs(run, k):
    out = {}
    for qid, doc_score_map in run.items():
        out[str(qid)] = [str(doc_id) for doc_id in list(doc_score_map.keys())[:k]]
    return out


def get_combos(models):
    combos = []
    for r in range(1, len(models) + 1):
        combos.extend(itertools.combinations(models, r))
    return combos


def union_topk_docs(combo, qid, model_runs):
    docs = []
    seen = set()
    for model in combo:
        for d in model_runs[model].get(qid, []):
            if d not in seen:
                seen.add(d)
                docs.append(d)
    return docs


def main():
    qrels_raw = load_json(QRELS_PATH)
    qrels = {
        str(qid): {str(doc_id) for doc_id, rel in doc_rel_map.items() if int(rel) > 0}
        for qid, doc_rel_map in qrels_raw.items()
    }

    model_runs = {}
    for model in MODELS:
        path = RUN_PATH_TEMPLATE.format(domain=DOMAIN, model=model)
        run = load_json(path)
        model_runs[model] = get_topk_docs(run, TOP_K)

    valid_qids = set(qrels.keys())
    for model in MODELS:
        valid_qids &= set(model_runs[model].keys())
    valid_qids = sorted(valid_qids)

    rows = []
    for combo in get_combos(MODELS):
        combo_name = "+".join(combo)

        total_positives = 0
        covered_positives = 0
        queries_with_hit = 0

        for qid in valid_qids:
            pos_docs = qrels[qid]
            retrieved_docs = set(union_topk_docs(combo, qid, model_runs))
            hit_docs = pos_docs & retrieved_docs

            total_positives += len(pos_docs)
            covered_positives += len(hit_docs)

            if len(hit_docs) > 0:
                queries_with_hit += 1

        n_queries = len(valid_qids)
        avg_positives_per_query = total_positives / n_queries if n_queries else 0.0
        coverage_at_50 = covered_positives / total_positives if total_positives else 0.0
        query_hit_rate = queries_with_hit / n_queries if n_queries else 0.0

        rows.append({
            "combination": combo_name,
            "n_queries": n_queries,
            "covered_positives_at_50": covered_positives,
            "coverage_at_50": round(coverage_at_50, 4),
            "queries_with_hit": queries_with_hit,
            "query_hit_rate": round(query_hit_rate, 4),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["coverage_at_50", "query_hit_rate"], ascending=False)

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

# python scripts/analysis/get_pool_coverage.py
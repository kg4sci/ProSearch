import json
from collections import defaultdict

def rrf_fuse(results_list):
    doc_scores = defaultdict(float)
    for res in results_list:
        ranked = sorted(res.keys(), key=lambda x: -res[x])
        for rank, docid in enumerate(ranked, 1):
            doc_scores[docid] += 1.0 / (RRF_K + rank)
    return dict(sorted(doc_scores.items(), key=lambda x: -x[1]))

DOMAINS = [
    "electrochemistry",
    # "cell_culture",
    # "immunotechnology",
    # "inorganic_materials",
    # "organic_synthesis",
    # "polymer_materials"
]

models = [
    # 'nv-embed',
    'bm25',
]

for domain in DOMAINS:
    for model in models:
        print(f"正在计算: Domain={domain}, Model={model}")
        EXPANDED_QUERIES = f"data/{domain}/gen_query/queries_decomp_expanded.json"  
        DECOMP_RESULTS = f"data/{domain}/eval/{model}/run_top100_decomp_expanded.json" 
        OUTPUT = f"data/{domain}/eval/{model}/run_top100_decomp.json"     
        RRF_K = 60

        expanded = json.load(open(EXPANDED_QUERIES, encoding="utf-8"))
        decomp_runs = json.load(open(DECOMP_RESULTS, encoding="utf-8"))

        parent_results = defaultdict(list)
        for subq_id, subq_data in expanded.items():
            parent_id = subq_data["parent_id"]
            if subq_id in decomp_runs:
                parent_results[parent_id].append(decomp_runs[subq_id])

        final = {pid: rrf_fuse(res_list) for pid, res_list in parent_results.items()}

        json.dump(final, open(OUTPUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# python scripts/eval/merge_decomp.py



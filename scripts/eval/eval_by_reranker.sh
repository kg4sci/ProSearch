#!/usr/bin/env bash

set -euo pipefail

# DOMAIN_LIST="electrochemistry,cell_culture,immunotechnology,inorganic_materials,organic_synthesis,polymer_materials" # 逗号分隔
DOMAIN_LIST="electrochemistry"
MODEL_LIST="bge_reranker,qwen3_reranker,rank1,rank_r1,reasonrank,diver_grouprank"

# "recip_rank": "MRR"
METRICS="ndcg_cut_10"
# METRICS="P_10,ndcg_cut_10,recall_10,recip_rank"
# GROUP_FIELDS="type,constraint_count,has_numeric,has_negation"
TYPE="raw"
OUTPUT_CSV="result/tables/all_metrics_by_reranker.csv"

python scripts/eval/eval.py \
  --domains "${DOMAIN_LIST}" \
  --models "${MODEL_LIST}" \
  --metrics "${METRICS}" \
  --output_csv "${OUTPUT_CSV}" \
  --queries_relpath "data/{domain}/gen_query/queries_raw.json" \
  --qrels_relpath "data/{domain}/stage1_relevance/qrels_stage1.json" \
  --run_relpath "data/{domain}/eval/{model}/run_top100_${TYPE}.json"

# conda activate retrieve_dataset
# cd /home/liangwenliang/检索数据集/

# --qrels_relpath "data/{domain}/stage1_relevance/qrels_stage1.json" \
# --qrels_relpath "data/{domain}/gen_query/qrels_llm.json" \

# --run_relpath "data/{domain}/eval/{model}/run_top100_${TYPE}.json"
# --run_relpath "data/{domain}/eval/{model}/run_top100_raw_gpt-4o-mini.json"
# bash scripts/eval/eval_by_reranker.sh

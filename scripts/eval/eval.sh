#!/usr/bin/env bash

set -euo pipefail

# DOMAIN_LIST="electrochemistry,cell_culture,immunotechnology,inorganic_materials,organic_synthesis,polymer_materials" # 逗号分隔
DOMAIN_LIST="electrochemistry"
# MODEL_LIST="bm25,specter2,sbert,gte_large,bge_reasoner,nv_embed"
# MODEL_LIST="bm25,sbert,harrier-oss-v1,bge_m3,,gte_large,nv_embed,qwen3-8b,specter2,scincl,pubmedbert-base,bioclinical-modernbert,ChEmbed-prog,chemical-bert,matscibert,bge_reasoner,reasonIR,reasonEmbed"
# MODEL_LIST="agentic"
# MODEL_LIST="inf-retriever-v1"
MODEL_LIST="diver_retriever"
# "recip_rank": "MRR"
METRICS="ndcg_cut_10"
# METRICS="P_10,ndcg_cut_10,recall_10,recip_rank"
# GROUP_FIELDS="type,constraint_count,has_numeric,has_negation"
GROUP_FIELDS=""
# TYPE="inf_query_aligner"
TYPE="diverqe_round2_qwen"
# TYPE="raw"
OUTPUT_CSV="result/tables/all_metrics_${TYPE}.csv"

python scripts/eval/eval.py \
  --domains "${DOMAIN_LIST}" \
  --models "${MODEL_LIST}" \
  --metrics "${METRICS}" \
  --group_fields "${GROUP_FIELDS}" \
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
# bash scripts/eval/eval.sh

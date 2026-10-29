#!/usr/bin/env bash

set -euo pipefail

# DOMAIN_LIST="electrochemistry,cell_culture,immunotechnology,inorganic_materials,organic_synthesis,polymer_materials" # 逗号分隔
DOMAIN_LIST="electrochemistry"
MODEL_LIST="bm25,nv_embed"
METRICS="ndcg_cut_10"

TYPES=(
    "raw"
    "hyde"
    "decomp"
    "reasoning_first"
    "reasoning"
    "structured"
    "inf_query_aligner"
    "tongsearch"
    "diverqe_round2_gpt"
)


OUT_DIR="result/tables"
TMP_DIR="${OUT_DIR}/tmp_eval_merge"
FINAL_CSV="${OUT_DIR}/all_metrics_by_query_type.csv"

mkdir -p "${OUT_DIR}"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

FIRST=1

for TYPE in "${TYPES[@]}"; do
  TMP_CSV="${TMP_DIR}/${TYPE}.csv"

  echo "Running TYPE=${TYPE}"

  python scripts/eval/eval.py \
    --domains "${DOMAIN_LIST}" \
    --models "${MODEL_LIST}" \
    --metrics "${METRICS}" \
    --output_csv "${TMP_CSV}" \
    --queries_relpath "data/{domain}/gen_query/queries_raw.json" \
    --qrels_relpath "data/{domain}/stage1_relevance/qrels_stage1.json" \
    --run_relpath "data/{domain}/eval/{model}/run_top100_${TYPE}.json"

  if [[ ! -f "${TMP_CSV}" ]]; then
    echo "Missing output file: ${TMP_CSV}"
    exit 1
  fi

  if [[ ${FIRST} -eq 1 ]]; then
    awk -v t="${TYPE}" 'NR==1{print "type," $0} NR>1{print t "," $0}' "${TMP_CSV}" > "${FINAL_CSV}"
    FIRST=0
  else
    awk -v t="${TYPE}" 'NR>1{print t "," $0}' "${TMP_CSV}" >> "${FINAL_CSV}"
  fi
done

echo "Merged csv saved to: ${FINAL_CSV}"
rm -rf "${TMP_DIR}"

# bash scripts/eval/eval_by_query_type.sh



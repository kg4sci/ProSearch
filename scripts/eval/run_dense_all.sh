#!/usr/bin/env bash
set -euo pipefail

DOMAINS=(
    "electrochemistry"
    "cell_culture"
    "immunotechnology"
    "inorganic_materials"
    "organic_synthesis"
    "polymer_materials"
)

MODELS=(
    # "specter2|model/specter2_base"
    # "gte_large|model/gte-large-en-v1.5"
    # "sbert|model/all-mpnet-base-v2"
    # "bge_reasoner|model/bge-reasoner"
    "nv_embed|model/NV-Embed-v2"
)


# bm25, dense
RETRIEVAL_MODE="dense"
TYPE='raw'
DEVICE="cuda"
CORPUS_TEXT_FIELD="paragraph"
TOP_K=100
BATCH_SIZE=4
SCORE_FUNCTION="dot"
OVERWRITE_CORPUS_EMBEDDINGS=false
RUN_NAME="dense_retrieval"

# --- 4. 嵌套循环开始 ---
for DOMAIN in "${DOMAINS[@]}"; do
    echo "===================================================="
    echo "Starting Domain: ${DOMAIN}"
    echo "===================================================="

    for entry in "${MODELS[@]}"; do
        MODEL_FAMILY="${entry%%|*}"
        MODEL_NAME_OR_PATH="${entry##*|}"

        echo ">>> Running [${DOMAIN}] with [${MODEL_FAMILY}]"

        # QUERIES_PATH="data/${DOMAIN}/gen_query/queries_${TYPE}.json"
        QUERIES_PATH="data/cell_culture/gen_query/queries_${TYPE}.json"

        CORPUS_PATH="seed_data/${DOMAIN}/all_paragraphs.json"
        CACHE_DIR="data/${DOMAIN}/eval/cache"
        OUTPUT_DIR="data/${DOMAIN}/eval"

        DENSE_RUN_JSON="${OUTPUT_DIR}/${MODEL_FAMILY}/all_run_top${TOP_K}_${TYPE}.json"
        DENSE_RUN_TREC="${OUTPUT_DIR}/${MODEL_FAMILY}/all_run_top${TOP_K}_${TYPE}.trec"
        
        BM25_RUN_JSON="${OUTPUT_DIR}/bm25/all_run_top${TOP_K}_${TYPE}.json"
        BM25_RUN_TREC="${OUTPUT_DIR}/bm25/all_run_top${TOP_K}_${TYPE}.trec"

        CORPUS_EMB_PATH="${CACHE_DIR}/${MODEL_FAMILY}/all_paragraph_emb.npy"
        CORPUS_DOC_IDS_PATH="${CACHE_DIR}/all_paragraph_doc_ids.json"

        # 确保输出目录存在
        mkdir -p "${OUTPUT_DIR}/${MODEL_FAMILY}"
        mkdir -p "${CACHE_DIR}/${MODEL_FAMILY}"

        # --- 执行推理 ---
        CUDA_VISIBLE_DEVICES=6,7 python scripts/eval/run_dense.py \
          --domain "${DOMAIN}" \
          --retrieval_mode "${RETRIEVAL_MODE}" \
          --model_family "${MODEL_FAMILY}" \
          --model_name_or_path "${MODEL_NAME_OR_PATH}" \
          --device "${DEVICE}" \
          --queries_path "${QUERIES_PATH}" \
          --corpus_path "${CORPUS_PATH}" \
          --corpus_text_field "${CORPUS_TEXT_FIELD}" \
          --cache_dir "${CACHE_DIR}" \
          --output_dir "${OUTPUT_DIR}" \
          --dense_run_json "${DENSE_RUN_JSON}" \
          --dense_run_trec "${DENSE_RUN_TREC}" \
          --bm25_run_json "${BM25_RUN_JSON}" \
          --bm25_run_trec "${BM25_RUN_TREC}" \
          --corpus_emb_path "${CORPUS_EMB_PATH}" \
          --corpus_doc_ids_path "${CORPUS_DOC_IDS_PATH}" \
          --batch_size "${BATCH_SIZE}" \
          --top_k "${TOP_K}" \
          --score_function "${SCORE_FUNCTION}" \
          --run_name "${RUN_NAME}" \
          $( [ "${OVERWRITE_CORPUS_EMBEDDINGS}" = true ] && echo "--overwrite_corpus_embeddings" )

        echo -e "Done: ${DOMAIN} - ${MODEL_FAMILY}\n"
    done
done

echo "All tasks completed!"

# conda activate nvembed
# conda activate retrieve_dataset
# cd /home/liangwenliang/检索数据集/
# bash scripts/eval/run_dense.sh

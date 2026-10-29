#!/usr/bin/env bash
set -euo pipefail

DOMAINS=(
  # "electrochemistry"
  # "cell_culture"
  # "immunotechnology"
  # "inorganic_materials"
  # "organic_synthesis"
  "polymer_materials"
)

API_KEYS_PATH="scripts/keys.json"
MAX_WORKERS=50
TEMPERATURE=0.0
MAX_TOKENS=4096
MAX_RETRIES=5
RETRY_SLEEP=2.0

MODE="supplement"
# MODE="overwrite"

SAVE_PROMPT_TEXT="false"
BASE_URL="https://one-api.bltcy.top/v1"
MODEL="gpt-5.2"

for domain in "${DOMAINS[@]}"; do
    CANDIDATE_PARAGRAPH_IDS_PATH="data/${domain}/relevance/pooling_top50.json"
    BOUNDARY_MANIFEST_PATH="data/${domain}/stage1_relevance/sampled_constraints_stage1.boundary_manifest.jsonl"

    PARAGRAPH_PATH="seed_data/${domain}/paragraphs.json"

    OUTPUT_JSONL_PATH="data/${domain}/relevance/judgments_top50.jsonl"
    OUTPUT_STATS_PATH="data/${domain}/relevance/judgments_top50.stats.json"

    CMD=(
        python scripts/relevance/judge.py
        --domain "$domain"
        --api_keys_path "$API_KEYS_PATH"
        --model "$MODEL"
        --max_workers "$MAX_WORKERS"
        --temperature "$TEMPERATURE"
        --max_tokens "$MAX_TOKENS"
        --max_retries "$MAX_RETRIES"
        --retry_sleep "$RETRY_SLEEP"
        --mode "$MODE"
        --candidate_paragraph_ids_path "$CANDIDATE_PARAGRAPH_IDS_PATH"
        --boundary_manifest_path "$BOUNDARY_MANIFEST_PATH"
        --paragraph_path "$PARAGRAPH_PATH"
        --output_jsonl_path "$OUTPUT_JSONL_PATH"
        --output_stats_path "$OUTPUT_STATS_PATH"
    )

    if [[ "$SAVE_PROMPT_TEXT" == "true" ]]; then
    CMD+=(--save_prompt_text)
    fi

    if [[ -n "$BASE_URL" ]]; then
    CMD+=(--base_url "$BASE_URL")
    fi

    printf 'Running command:\n%s\n' "${CMD[*]}"
    "${CMD[@]}"

done

# conda activate retrieve_dataset
# bash scripts/relevance/judge.sh


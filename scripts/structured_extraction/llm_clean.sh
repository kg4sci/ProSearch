#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://one-api.bltcy.top/v1"
MODEL_NAME="gpt-5.2"
KEYS_JSON="scripts/keys.json"
WORKERS=50
MAX_TOKENS=4096
TEMPERATURE=0.0
TIMEOUT=120
MAX_RETRIES=5
SLEEP_SECONDS=2
BATCH_SIZE=50

DOMAINS=(
  "electrochemistry"
  "cell_culture"
  "immunotechnology"
  "inorganic_materials"
  "organic_synthesis"
  # "polymer_materials"
)

for DOMAIN in "${DOMAINS[@]}"; do
  INPUT_PATH="seed_data/${DOMAIN}/llm_extraction/field_values.json"
  OUTPUT_PATH="seed_data/${DOMAIN}/llm_cleaning/field_values_stage1.json"

  # INPUT_PATH="seed_data/${DOMAIN}/llm_cleaning/field_values_stage1.json"
  # OUTPUT_PATH="seed_data/${DOMAIN}/llm_cleaning/field_values_stage2.json"

  python scripts/structured_extraction/llm_clean.py \
    --domain "$DOMAIN" \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --keys-json "$KEYS_JSON" \
    --workers "$WORKERS" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --timeout "$TIMEOUT" \
    --max-retries "$MAX_RETRIES" \
    --sleep-seconds "$SLEEP_SECONDS" \
    --test false \
    --test-slot "light_source_type" \
    --max-batches-per-slot 1
done


# cd /home/liangwenliang/检索数据集/
# conda activate retrieve_dataset
# bash scripts/structured_extraction/llm_clean.sh


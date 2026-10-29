#!/usr/bin/env bash

set -euo pipefail

DOMAINS=(
  "electrochemistry"
#   "cell_culture"
#   "immunotechnology"
#   "inorganic_materials"
#   "organic_synthesis"
#   "polymer_materials"
)

TASK_TYPES=(
  "gen"
  "eval"
)

BASE_URL="https://one-api.bltcy.top/v1"
KEYS_PATH="scripts/keys.json"

MODE="overwrite"
# MODE="supplement"

MAX_RETRIES=5
SLEEP_SECONDS=2
GEN_TEMPERATURE=0.7
EVAL_TEMPERATURE=0

for domain in "${DOMAINS[@]}"; do
  mkdir -p "data/${domain}/gen_query"

  sampled_input="data/${domain}/manual_audit/filtered_boundary_manifest.jsonl"
  gen_output="data/${domain}/manual_audit/generated_queries.jsonl"
  eval_output="data/${domain}/manual_audit/evaluated_queries.jsonl"

  for task_type in "${TASK_TYPES[@]}"; do
    case "$task_type" in
      gen)
        MODEL="gpt-5.2"
        input_path="$sampled_input"
        output_path="$gen_output"
        ;;
      eval)
        MODEL="gemini-2.5-pro"
        input_path="$gen_output"
        output_path="$eval_output"
        ;;
      *)
        echo "Unknown task_type: $task_type" >&2
        exit 1
        ;;
    esac

    echo "=================================================="
    echo "Running domain=${domain}, task_type=${task_type}"
    echo "model=${MODEL}"
    echo "input_path=${input_path}"
    echo "output_path=${output_path}"
    echo "mode=${MODE}"
    echo "=================================================="

    python scripts/gen_query/gen_query.py \
      --domain "$domain" \
      --model "$MODEL" \
      --task_type "$task_type" \
      --input_path "$input_path" \
      --output_path "$output_path" \
      --keys_path "$KEYS_PATH" \
      --base_url "$BASE_URL" \
      --mode "$MODE" \
      --max_retries "$MAX_RETRIES" \
      --sleep_seconds "$SLEEP_SECONDS" \
      --gen_temperature "$GEN_TEMPERATURE" \
      --eval_temperature "$EVAL_TEMPERATURE" \
      --num_workers 40
  done
done

# cd /home/liangwenliang/检索数据集
# conda activate retrieve_dataset
# bash scripts/manual_audit/gen_query.sh



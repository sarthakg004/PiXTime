#!/bin/bash

# Audit findings:
# - This repo already uses dataset names ETTh1/ETTh2/ETTm1/ETTm2/electricity/exchange_rate/traffic/weather.
# - Canonical horizons in existing run scripts are 96, 192, 336, 720.
# - Baseline PiXTime hyperparameters vary by dataset/horizon and are preserved below.

set -e

CONFIGS=("baseline" "contextual_ve" "var_relation" "adaptive_patch" "multiscale_patch" "all_improvements")
PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-10}
RESULTS_CSV=${RESULTS_CSV:-results/enhanced_results.csv}

is_done () {
  local DATA=$1
  local PRED_LEN=$2
  local CONFIG=$3

  if [[ ! -f "$RESULTS_CSV" ]]; then
    return 1
  fi

  awk -F',' -v d="$DATA" -v p="$PRED_LEN" -v c="$CONFIG" '
    NR > 1 {
      gsub(/\r/, "", $0)
      if ($1 == d && $2 == p && $3 == c) {
        found = 1
      }
    }
    END { exit(found ? 0 : 1) }
  ' "$RESULTS_CSV"
}

run_block () {
  local DATA=$1
  local PRED_LEN=$2
  local D_MODEL=$3
  local EN_D_FF=$4
  local DE_D_FF=$5
  local EN_LAYERS=$6
  local DE_LAYERS=$7
  local ROOT_ARG=$8

  for config in "${CONFIGS[@]}"; do
    EXTRA_ARGS=()
    MODEL="PiXTime_Enhanced"

    case $config in
      baseline)
        MODEL="PiXTime"
        ;;
      contextual_ve)
        EXTRA_ARGS+=("--use_contextual_var_emb")
        ;;
      var_relation)
        EXTRA_ARGS+=("--use_var_relation")
        ;;
      adaptive_patch)
        EXTRA_ARGS+=("--use_adaptive_patch")
        ;;
      multiscale_patch)
        EXTRA_ARGS+=("--use_multiscale_patch")
        ;;
      all_improvements)
        EXTRA_ARGS+=("--use_contextual_var_emb" "--use_var_relation" "--use_adaptive_patch" "--use_multiscale_patch")
        ;;
    esac

    if is_done "$DATA" "$PRED_LEN" "$config"; then
      echo "Skipping DATA=${DATA}, pred_len=${PRED_LEN}, config=${config} (already in ${RESULTS_CSV})"
      continue
    fi

    echo "Running DATA=${DATA}, pred_len=${PRED_LEN}, config=${config}"

    ${PYTHON_BIN} run.py \
      --model ${MODEL} \
      --data ${DATA} \
      --features M \
      --target OT \
      --seq_len 96 \
      --pred_len ${PRED_LEN} \
      --train_epochs ${TRAIN_EPOCHS} \
      --d_model ${D_MODEL} \
      --en_d_ff ${EN_D_FF} \
      --de_d_ff ${DE_D_FF} \
      --en_layers ${EN_LAYERS} \
      --de_layers ${DE_LAYERS} \
      ${ROOT_ARG} \
      "${EXTRA_ARGS[@]}" \
      --result_tag ${config}
  done
}

# ===== ETTh1 =====
run_block "ETTh1" 96 512 2048 2048 1 1 ""
run_block "ETTh1" 192 512 2048 2048 1 1 ""
run_block "ETTh1" 336 512 2048 2048 1 1 ""
run_block "ETTh1" 720 128 512 512 1 1 ""

# ===== ETTh2 =====
run_block "ETTh2" 96 256 1024 1024 1 1 ""
run_block "ETTh2" 192 128 512 512 1 1 ""
run_block "ETTh2" 336 128 512 512 1 1 ""
run_block "ETTh2" 720 512 2048 2048 1 1 ""

# ===== ETTm1 =====
run_block "ETTm1" 96 512 2048 2048 1 1 ""
run_block "ETTm1" 192 512 2048 2048 1 1 ""
run_block "ETTm1" 336 512 2048 2048 1 1 ""
run_block "ETTm1" 720 512 2048 2048 1 1 ""

# ===== ETTm2 =====
run_block "ETTm2" 96 512 2048 2048 1 1 ""
run_block "ETTm2" 192 256 1024 1024 1 1 ""
run_block "ETTm2" 336 256 1024 1024 1 1 ""
run_block "ETTm2" 720 256 1024 1024 1 1 ""

# ===== electricity =====
run_block "electricity" 96 512 2048 2048 2 2 "--root_path ./dataset/electricity/"
run_block "electricity" 192 512 2048 2048 2 2 "--root_path ./dataset/electricity/"
run_block "electricity" 336 512 2048 2048 2 2 "--root_path ./dataset/electricity/"
run_block "electricity" 720 512 2048 2048 2 2 "--root_path ./dataset/electricity/"

# ===== exchange_rate =====
run_block "exchange_rate" 96 64 256 256 1 1 "--root_path ./dataset/exchange_rate/"
run_block "exchange_rate" 192 64 256 256 1 1 "--root_path ./dataset/exchange_rate/"
run_block "exchange_rate" 336 64 256 256 1 1 "--root_path ./dataset/exchange_rate/"
run_block "exchange_rate" 720 128 512 512 1 1 "--root_path ./dataset/exchange_rate/"

# ===== traffic =====
run_block "traffic" 96 512 2048 2048 1 1 "--root_path ./dataset/traffic/"
run_block "traffic" 192 512 2048 2048 1 1 "--root_path ./dataset/traffic/"
run_block "traffic" 336 512 2048 2048 1 1 "--root_path ./dataset/traffic/"
run_block "traffic" 720 512 2048 2048 1 1 "--root_path ./dataset/traffic/"

# ===== weather =====
run_block "weather" 96 512 2048 2048 1 1 "--root_path ./dataset/weather/"
run_block "weather" 192 512 2048 2048 1 1 "--root_path ./dataset/weather/"
run_block "weather" 336 512 2048 2048 1 1 "--root_path ./dataset/weather/"
run_block "weather" 720 512 2048 2048 1 1 "--root_path ./dataset/weather/"

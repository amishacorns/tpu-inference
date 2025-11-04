#!/usr/bin/env bash
# Script to run baseline vs quantized KL/entropy comparison using compare_kl_api.py
# - Starts a vLLM server for baseline (no quant), dumps top-K prompt_logprobs
# - Iterates through quant configurations, starts server, compares against baseline
# - Logs all outputs and writes a CSV summary with mean_kl and mean_entropy

set -euo pipefail

# --- Configuration ---
MODELS=("openai/gpt-oss-20b")
TENSOR_PARALLEL_SIZE=8
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=1             # For stable prompt_logprobs dumps, uncertain on correctness with multiple sequences
K=256                      # Top-K for prompt_logprobs
MAX_PROMPTS=0              # Cap prompts per model for KL runs

# Dataset config for compare_kl_api.py
DATASET="wikitext"
DATASET_NAME="wikitext-2-raw-v1"
SPLIT="test"
# Keep a small headroom vs model max (avoid BOS/overhead overruns)
CTX_LEN=$((MAX_MODEL_LEN-1))
STRIDE=0
MAX_EVAL_TOKENS=0         # 0 = unlimited

# Paths
OUT_DIR="/workspace/kl_results"
DUMPS_DIR="${OUT_DIR}/dumps"
RESULTS_DIR="${OUT_DIR}/results"
LOGS_DIR="${OUT_DIR}/logs"
SUMMARIES_DIR="${OUT_DIR}/summaries"
SUMMARY_CSV="${SUMMARIES_DIR}/kl_summary.csv"

# Server/health
VLLM_PORT=${VLLM_PORT:-8000}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-1800}
SHUTDOWN_SLEEP=${SHUTDOWN_SLEEP:-30}

# Quantization config sweep
MODULE_PATHS=(".*custom_module")
QTYPE_TUPLES=(
  "float4_e2m1fn:float8_e4m3fn"
  "float8_e4m3fn:float8_e4m3fn"
)
TILE_SIZES=(32 64 128 256 512)

# --- Environment ---
export TPU_BACKEND_TYPE=jax

# --- Helpers ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_vllm.sh
source "${SCRIPT_DIR}/lib_vllm.sh"

# --- Setup output dirs ---
mkdir -p "${DUMPS_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}" "${SUMMARIES_DIR}"
echo "📁 Output directory: ${OUT_DIR}"

# Init summary CSV (idempotent)
if [ ! -f "${SUMMARY_CSV}" ]; then
  echo "model,qtype_name,module_name,tile_size,mean_kl,mean_entropy,results_file,server_log,kl_log" > "${SUMMARY_CSV}"
fi

# --- Functions (reuse generic helpers from lib_vllm.sh) ---
# vllm_qtype_name, vllm_sanitize_module_name

compute_summary_csv_row() {
  local results_file="$1"
  python - "$results_file" <<'PY'
import json, sys, gzip, os
path = sys.argv[1]
open_func = gzip.open if path.endswith('.gz') else open
sum_w_kl = 0.0
sum_w_ent = 0.0
total_pos = 0
with open_func(path, 'rt', encoding='utf-8') as f:
  for line in f:
    line=line.strip()
    if not line:
      continue
    try:
      rec = json.loads(line)
    except Exception:
      continue
    pos = int(rec.get('positions') or 0)
    avg_kl = rec.get('avg_kl')
    avg_ent = rec.get('avg_entropy')
    if pos > 0 and avg_kl is not None and avg_ent is not None:
      sum_w_kl += avg_kl * pos
      sum_w_ent += avg_ent * pos
      total_pos += pos
if total_pos == 0:
  print(',,')
else:
  print(f"{sum_w_kl/total_pos},{sum_w_ent/total_pos}")
PY
}

# --- Main loop ---
for MODEL in "${MODELS[@]}"; do
  MODEL_SAFE=$(echo "$MODEL" | sed 's#/#_#g')

  echo ""; echo "════════════════════════════════════════════════════════════";
  echo "MODEL: ${MODEL}";
  echo "════════════════════════════════════════════════════════════";

  # Baseline: no quantization
  echo ""; echo "▶ Baseline (no quant): starting server...";
  kill_vllm_server

  BASELINE_SUFFIX="${MODEL_SAFE}_baseline_k${K}"
  BASELINE_DUMP="${DUMPS_DIR}/${BASELINE_SUFFIX}.jsonl.gz"
  BASELINE_SRV_LOG="${LOGS_DIR}/server_${BASELINE_SUFFIX}.log"
  BASELINE_RUN_LOG="${LOGS_DIR}/kl_run_${BASELINE_SUFFIX}.log"

  vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --disable-log-requests \
    --max-logprobs "${K}" \
    2>&1 | tee "${BASELINE_SRV_LOG}" &
  VLLM_PID=$!
  echo "   Server PID: $VLLM_PID"

  if ! wait_for_server; then
    echo "❌ Failed to start baseline server for ${MODEL}"
    echo "Last 50 lines of server log:"; tail -50 "${BASELINE_SRV_LOG}" || true
    kill_vllm_server
    continue
  fi

  echo "📥 Dumping baseline top-K prompt_logprobs to ${BASELINE_DUMP}"
  # Allow unlimited prompts when MAX_PROMPTS<=0 by omitting the flag
  if [[ "${MAX_PROMPTS}" -gt 0 ]]; then
    PROMPTS_ARGS=(--max-prompts "${MAX_PROMPTS}")
  else
    PROMPTS_ARGS=()
  fi
  python "${SCRIPT_DIR}/compare_kl_api.py" \
    --model "${MODEL}" \
    --base-url "http://localhost:${VLLM_PORT}/v1/completions" \
    --k "${K}" \
    "${PROMPTS_ARGS[@]}" \
    --dataset "${DATASET}" \
    --dataset-name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --ctx-len "${CTX_LEN}" \
    --stride "${STRIDE}" \
    --max-eval-tokens "${MAX_EVAL_TOKENS}" \
    --dump "${BASELINE_DUMP}" \
    | tee "${BASELINE_RUN_LOG}"

  echo "🛑 Shutting down baseline server..."
  kill $VLLM_PID 2>/dev/null || true
  echo "⏳ Waiting ${SHUTDOWN_SLEEP}s for graceful shutdown..."; sleep "${SHUTDOWN_SLEEP}"
  kill_vllm_server

  # Quantized runs
  for QTYPE_TUPLE in "${QTYPE_TUPLES[@]}"; do
    WEIGHT_QTYPE="${QTYPE_TUPLE%%:*}"
    ACT_QTYPE="${QTYPE_TUPLE##*:}"
  QNAME=$(vllm_qtype_name "$WEIGHT_QTYPE" "$ACT_QTYPE")

    for MODULE_PATH in "${MODULE_PATHS[@]}"; do
  MNAME=$(vllm_sanitize_module_name "$MODULE_PATH")

      for TILE_SIZE in "${TILE_SIZES[@]}"; do
        echo ""; echo "▶ Quant run: ${MODEL} | ${QNAME} | ${MNAME} | tile=${TILE_SIZE}"
        kill_vllm_server

        SUFFIX="${MODEL_SAFE}_${QNAME}_${MNAME}_tile_${TILE_SIZE}_k${K}"
        SrvLog="${LOGS_DIR}/server_${SUFFIX}.log"
        RunLog="${LOGS_DIR}/kl_run_${SUFFIX}.log"
        ResultFile="${RESULTS_DIR}/${SUFFIX}_kl_results.jsonl.gz"

        if [[ "$QNAME" == "baseline" ]]; then
          ADDITIONAL_CONFIG_ARGS=()
        else
          ADDITIONAL_CONFIG='{"quantization": {"qwix": {"rules": [{"module_path": "'${MODULE_PATH}'", "weight_qtype": "'${WEIGHT_QTYPE}'", "act_qtype": "'${ACT_QTYPE}'", "tile_size": '${TILE_SIZE}'}]}}, "is_verbose": true}'
          ADDITIONAL_CONFIG_ARGS=(--additional-config "${ADDITIONAL_CONFIG}")
        fi

        echo "📝 Starting server (log: ${SrvLog})..."
        vllm serve "${MODEL}" \
          --port "${VLLM_PORT}" \
          --max_model_len "${MAX_MODEL_LEN}" \
          --max-num-seqs "${MAX_NUM_SEQS}" \
          --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
          --disable-log-requests \
          --max-logprobs "${K}" \
          "${ADDITIONAL_CONFIG_ARGS[@]}" \
          2>&1 | tee "${SrvLog}" &
        VLLM_PID=$!
        echo "   Server PID: $VLLM_PID"

        if ! wait_for_server; then
          echo "❌ Failed to start server for ${SUFFIX}"
          echo "Last 50 lines of server log:"; tail -50 "${SrvLog}" || true
          kill_vllm_server
          continue
        fi

        echo "📊 Running KL compare (save results to ${ResultFile})"
        # Allow unlimited prompts when MAX_PROMPTS<=0 by omitting the flag
        if [[ "${MAX_PROMPTS}" -gt 0 ]]; then
          PROMPTS_ARGS=(--max-prompts "${MAX_PROMPTS}")
        else
          PROMPTS_ARGS=()
        fi
        python "${SCRIPT_DIR}/compare_kl_api.py" \
          --model "${MODEL}" \
          --base-url "http://localhost:${VLLM_PORT}/v1/completions" \
          --k "${K}" \
          "${PROMPTS_ARGS[@]}" \
          --dataset "${DATASET}" \
          --dataset-name "${DATASET_NAME}" \
          --split "${SPLIT}" \
          --ctx-len "${CTX_LEN}" \
          --stride "${STRIDE}" \
          --max-eval-tokens "${MAX_EVAL_TOKENS}" \
          --load "${BASELINE_DUMP}" \
          --save "${ResultFile}" \
          | tee "${RunLog}"

        echo "🛑 Shutting down server..."
        kill $VLLM_PID 2>/dev/null || true
        echo "⏳ Waiting ${SHUTDOWN_SLEEP}s for graceful shutdown..."; sleep "${SHUTDOWN_SLEEP}"
        kill_vllm_server

        # Compute and append summary CSV row
        read -r MEAN_KL MEAN_ENTROPY < <(compute_summary_csv_row "${ResultFile}" | tr ',' ' ')
        echo "$(printf '%s,%s,%s,%s,%s,%s,%s,%s,%s' \
          "$MODEL" "$QNAME" "$MNAME" "$TILE_SIZE" "${MEAN_KL}" "${MEAN_ENTROPY}" "${ResultFile}" "${SrvLog}" "${RunLog}")" >> "${SUMMARY_CSV}"
        echo "✅ Appended to summary: mean_kl=${MEAN_KL:-NA}, mean_entropy=${MEAN_ENTROPY:-NA}"
      done
      echo "└─ Completed module: ${MNAME}"
    done
    echo "┗━ Completed qtype: ${QNAME}"
  done
  echo "╚═ Completed model: ${MODEL}"

done

echo ""; echo "🎉 KL evaluations completed!"
echo "📁 Artifacts:"
echo "  Dumps:     ${DUMPS_DIR}/*.jsonl.gz"
echo "  Results:   ${RESULTS_DIR}/*_kl_results.jsonl.gz"
echo "  Logs:      ${LOGS_DIR}/*.log"
echo "  Summary:   ${SUMMARY_CSV}"
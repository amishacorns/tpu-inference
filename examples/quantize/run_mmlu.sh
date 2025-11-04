#!/bin/bash
# Script to sweep tile sizes for vllm quantization and run MMLU benchmark
# Usage: bash run_eval.sh [--enable-quantization]

set -e  # Exit on any error

# Configuration
MODELS=("openai/gpt-oss-20b")  # Models to evaluate
TENSOR_PARALLEL_SIZE=8
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=1024
# VLLM server knobs
GPU_MEMORY_UTILIZATION=0.8   # e.g., 0.8 for safety, 0.95 for higher packing
SERVE_EXTRA_ARGS="--disable-log-stats --no-enable-prefix-caching"          # e.g., "--disable-log-stats --no-enable-prefix-caching"
MAX_NUM_BATCHED_TOKENS=4096  # cap total batched tokens across all sequences
DATASET_PATH="/workspace/mmlu/data/test/"
OUT_DIR="/workspace/eval_results_absmax_calibration"  # Output directory for logs and results
NUM_PROMPTS=14042
MMLU_NUM_SHOTS=0
# MMLU sequence lengths
MMLU_INPUT_LEN=1536
MMLU_OUTPUT_LEN=2048
# MMLU formatting
MMLU_USE_CHAT_TEMPLATE=true
VLLM_PORT=8000
HEALTH_CHECK_TIMEOUT=1800  # 30 minutes max wait

# Timing Configuration
SHUTDOWN_SLEEP=30         # Time to wait after killing server (seconds)

# Quantization Configuration
MODULE_PATHS=(".*custom_module")  # Module patterns to quantize (matches paths with both mlp and proj)

# Calibration method: "absmax" or "mse,start,end,steps" (e.g., "mse,0.5,1.0,21")
CALIBRATION_METHOD="absmax"

# Quantization type tuples: "weight_qtype:act_qtype" (use "none:none" to disable quantization)
QTYPE_TUPLES=(
    "float4_e2m1fn:float8_e4m3fn"   # FP4 weights, FP8 activations
    "float8_e4m3fn:float8_e4m3fn"   # FP8 weights, FP8 activations
    # "float4_e2m1fn:float4_e2m1fn"   # FP4 weights, FP4 activations
)

TILE_SIZES=(
    32
    64
    128
    256
    512
) # Tile sizes to sweep

# Environment variables
# export NEW_MODEL_DESIGN=True
export TPU_BACKEND_TYPE=jax

# Shared helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Sourcing lib_vllm.sh from ${SCRIPT_DIR}/lib_vllm.sh"
# shellcheck source=lib_vllm.sh
source "${SCRIPT_DIR}/lib_vllm.sh"

# Main loop
# Create output directory if it doesn't exist
mkdir -p "${OUT_DIR}"
echo "📁 Output directory: ${OUT_DIR}"

# Initial cleanup
echo "🧹 Initial cleanup..."
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "python.*vllm" 2>/dev/null || true
rm -f /tmp/libtpu_lockfile 2>/dev/null || true
sleep 2

echo "⏳ Waiting for port ${VLLM_PORT} to be free..."
while lsof -i:${VLLM_PORT} > /dev/null 2>&1; do 
    sleep 2
done

echo "⏳ Waiting 10s for TPU to fully release..."
sleep 10
echo "✅ Cleanup complete"
echo ""

# Loop over models
for MODEL in "${MODELS[@]}"; do
    # Sanitize model name for filenames (replace / with _)
    MODEL_SAFE=$(echo "$MODEL" | sed 's/\//_/g')
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  MODEL: $MODEL"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # Loop over quantization type tuples
    for QTYPE_TUPLE in "${QTYPE_TUPLES[@]}"; do
        # Split tuple into weight and activation qtypes
        WEIGHT_QTYPE="${QTYPE_TUPLE%%:*}"
        ACT_QTYPE="${QTYPE_TUPLE##*:}"
        
        # Create readable name for qtype combo (generic)
        QTYPE_NAME=$(vllm_qtype_name "$WEIGHT_QTYPE" "$ACT_QTYPE")
        
        echo ""
        echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
        echo "┃  QTYPE: Weight=$WEIGHT_QTYPE, Act=$ACT_QTYPE ($QTYPE_NAME)"
        echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        echo ""

        # Loop over module paths
        for MODULE_PATH in "${MODULE_PATHS[@]}"; do
            # Create readable name for module path (generic)
            MODULE_NAME=$(vllm_sanitize_module_name "$MODULE_PATH")
            
            echo ""
            echo "┌────────────────────────────────────────────────────────────────┐"
            echo "│  MODULE PATH: $MODULE_PATH ($MODULE_NAME)"
            echo "└────────────────────────────────────────────────────────────────┘"
            echo ""

            # Loop over tile sizes
            for TILE_SIZE in "${TILE_SIZES[@]}"; do
                echo ""
                echo "============================================"
                echo "🚀 Model: $MODEL"
                echo "   QTypes: $QTYPE_NAME (W:$WEIGHT_QTYPE, A:$ACT_QTYPE)"
                echo "   Module: $MODULE_NAME ($MODULE_PATH)"
                echo "   Tile size: $TILE_SIZE"
                SUFFIX="${MODEL_SAFE}_${QTYPE_NAME}_${MODULE_NAME}_tile_${TILE_SIZE}"
                echo "Time: $(date)"
                echo "============================================"
    
                # Clean up any existing servers
                kill_vllm_server
                
                # Configure quantization (skip if none:none)
                if [[ "$QTYPE_NAME" == "baseline" ]]; then
                    ADDITIONAL_CONFIG_ARGS=()
                    echo "   Quantization: disabled (baseline)"
                else
                    ADDITIONAL_CONFIG='{"quantization": {"qwix": {"rules": [{"module_path": "'${MODULE_PATH}'", "weight_qtype": "'${WEIGHT_QTYPE}'", "act_qtype": "'${ACT_QTYPE}'", "tile_size": '$TILE_SIZE', "weight_calibration_method": "'${CALIBRATION_METHOD}'"}]}}, "is_verbose": true}'
                    ADDITIONAL_CONFIG_ARGS=(--additional-config "$ADDITIONAL_CONFIG")
                    echo "   Quantization: enabled"
                    echo "   Calibration method: ${CALIBRATION_METHOD}"
                fi
    
                LOG_FILE="${OUT_DIR}/vllm_server_${SUFFIX}.log"
                
                echo "📝 Starting VLLM server (log: $LOG_FILE)..."
                echo "   Model: ${MODEL}"
                echo "   Tensor Parallel: ${TENSOR_PARALLEL_SIZE}"
                if [[ "$QTYPE_NAME" != "baseline" ]]; then
                    echo "   Additional config: $ADDITIONAL_CONFIG"
                fi
                echo ""
                echo "🔍 Server output (also logging to $LOG_FILE):"
                echo "────────────────────────────────────────────────────────────────"
    
                # Start vllm serve in background with tee for real-time output
                vllm serve "${MODEL}" \
                    --max_model_len=${MAX_MODEL_LEN} \
                    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
                    --max-num-seqs=${MAX_NUM_SEQS} \
                    --disable-log-requests \
                    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
                    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
                    ${SERVE_EXTRA_ARGS} \
                    "${ADDITIONAL_CONFIG_ARGS[@]}" \
                    2>&1 | tee "$LOG_FILE" &
                
                VLLM_PID=$!
                echo "   Server PID: $VLLM_PID"
                
                # Wait for server to be ready
                if ! wait_for_server; then
                    echo "❌ Failed to start server"
                    echo ""
                    echo "📋 Last 50 lines of server log:"
                    echo "════════════════════════════════════════════════════════════════"
                    tail -50 "$LOG_FILE"
                    echo "════════════════════════════════════════════════════════════════"
                    echo ""
                    echo "💾 Full log saved to: $LOG_FILE"
                    kill_vllm_server
                    continue
                fi
    
                # Run MMLU benchmark
                echo ""
                echo "📊 Running MMLU benchmark..."
                RESULT_FILE="${OUT_DIR}/mmlu_results_${SUFFIX}.txt"
                # Conditionally add chat template flag
                if [[ "${MMLU_USE_CHAT_TEMPLATE}" == "1" || "${MMLU_USE_CHAT_TEMPLATE,,}" == "true" ]]; then
                    MMLU_CHAT_FLAG="--mmlu-use-chat-template"
                else
                    MMLU_CHAT_FLAG=""
                fi
                
                python scripts/vllm/benchmarking/benchmark_serving.py \
                    --backend vllm \
                    --model "${MODEL}" \
                    --dataset-name mmlu \
                    --dataset-path="${DATASET_PATH}" \
                    --num-prompts ${NUM_PROMPTS} \
                    --mmlu-num-shots ${MMLU_NUM_SHOTS} \
                    --mmlu-input-len ${MMLU_INPUT_LEN} \
                    --mmlu-output-len ${MMLU_OUTPUT_LEN} \
                    ${MMLU_CHAT_FLAG} \
                    --run-eval \
                    | tee "$RESULT_FILE"
                
                # Shutdown server
                echo ""
                echo "🛑 Shutting down VLLM server (PID $VLLM_PID)..."
                kill $VLLM_PID 2>/dev/null || true
                
                echo "⏳ Waiting ${SHUTDOWN_SLEEP}s for graceful shutdown..."
                sleep ${SHUTDOWN_SLEEP}
                
                # Force kill any remaining processes (especially EngineCore subprocess)
                pkill -9 -f "VLLM::EngineCore" || true
                pkill -9 -f "vllm serve" || true
                pkill -9 -f "python.*vllm" || true
                sleep 2
                
                # Wait for port to be completely free
                echo "⏳ Waiting for port ${VLLM_PORT} to be free..."
                while lsof -i:${VLLM_PORT} > /dev/null 2>&1; do 
                    sleep 2
                done
                
                # Clean up TPU lockfile
                if [ -f /tmp/libtpu_lockfile ]; then
                    echo "🧹 Removing TPU lockfile..."
                    rm -f /tmp/libtpu_lockfile || true
                fi
                
                echo "============================================"
                echo "✅ Completed: $MODEL / $QTYPE_NAME / $MODULE_NAME / tile_size=$TILE_SIZE"
                echo "   Results: $RESULT_FILE"
                echo "   Server log: $LOG_FILE"
                echo "============================================"
            done  # tile sizes
            
            echo ""
            echo "└─ Completed module: $MODULE_NAME"
            echo ""
        done  # module paths
        
        echo ""
        echo "┗━ Completed qtype: $QTYPE_NAME"
        echo ""
    done  # qtype tuples
    
    echo ""
    echo "╚═ Completed model: $MODEL"
    echo ""
done  # models

echo ""
echo "🎉 All evaluations completed!"
echo "📁 All results saved to: ${OUT_DIR}"
echo ""
echo "Summary:"
echo "  Models: ${MODELS[*]}"
echo "  QType tuples: ${QTYPE_TUPLES[*]}"
echo "  Module paths: ${MODULE_PATHS[*]}"
echo "  Tile sizes: ${TILE_SIZES[*]}"
echo "  Total runs: $((${#MODELS[@]} * ${#QTYPE_TUPLES[@]} * ${#MODULE_PATHS[@]} * ${#TILE_SIZES[@]}))"
echo ""
echo "  Results pattern: ${OUT_DIR}/*_tile_*.txt"
echo "  Logs pattern: ${OUT_DIR}/vllm_server_*_tile_*.log"
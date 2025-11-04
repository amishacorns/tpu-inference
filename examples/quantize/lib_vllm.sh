#!/usr/bin/env bash
# Shared helpers for vLLM server lifecycle on TPU
# Source this file from scripts in the same folder.

# Requires env vars (with defaults):
#   VLLM_PORT (default 8000)
#   HEALTH_CHECK_TIMEOUT (default 600 seconds)

wait_for_server() {
  local port="${VLLM_PORT:-8000}"
  local timeout="${HEALTH_CHECK_TIMEOUT:-600}"
  local max_attempts=$((timeout / 5))
  local attempt=0

  echo "🔍 Waiting for VLLM server to be ready at http://localhost:${port}/health..."
  while [ $attempt -lt $max_attempts ]; do
    if curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1; then
      echo "✅ Server is ready!"
      return 0
    fi

    if [ -n "${VLLM_PID:-}" ] && ! ps -p "$VLLM_PID" > /dev/null 2>&1; then
      echo "❌ Server process died!"
      return 1
    fi

    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts..."
    sleep 5
  done

  echo "❌ Server failed to become ready within ${timeout} seconds"
  return 1
}

kill_vllm_server() {
  local port="${VLLM_PORT:-8000}"

  echo "🔍 Checking for existing vllm/python processes..."

  # Kill the EngineCore subprocess first (this is what holds the TPU)
  if pgrep -f "VLLM::EngineCore" > /dev/null; then
    echo "⚠️  Killing VLLM::EngineCore subprocess..."
    pkill -9 -f "VLLM::EngineCore" || true
    sleep 2
  fi

  if pgrep -f "vllm serve" > /dev/null; then
    echo "⚠️  Killing vllm serve processes..."
    pkill -9 -f "vllm serve" || true
    sleep 2
  fi

  # Also kill any remaining python processes that might be holding TPU
  if pgrep -f "python.*vllm" > /dev/null; then
    echo "⚠️  Killing vllm-related python processes..."
    pkill -9 -f "python.*vllm" || true
    sleep 2
  fi

  # Clean up TPU lockfile
  if [ -f /tmp/libtpu_lockfile ]; then
    echo "🧹 Removing stale TPU lockfile..."
    rm -f /tmp/libtpu_lockfile 2>/dev/null || true
  fi

  # Wait for port to be free
  echo "⏳ Waiting for port ${port} to be free..."
  local wait_count=0
  while lsof -i:"${port}" > /dev/null 2>&1; do
    sleep 2
    wait_count=$((wait_count + 1))
    if [ $wait_count -gt 30 ]; then
      echo "⚠️  Port still busy after 60s, forcing..."
      fuser -k "${port}"/tcp 2>/dev/null || true
      sleep 2
      break
    fi
  done
  echo "✅ Port ${port} is free"

  # Extra sleep to let TPU fully release
  echo "⏳ Waiting 10s for TPU to fully release..."
  sleep 10
}

# --- Naming helpers (generic, non-hardcoded) ---
# Convert a module path regex to a readable, stable name.
# Example: ".*mlp.*proj.*" -> "mlp_proj", ".*proj.*" -> "proj", ".*" -> "all"
vllm_sanitize_module_name() {
  local raw="$1"
  # Replace any non-alphanumeric character with underscore, trim underscores, lowercase
  local cleaned
  cleaned=$(echo -n "$raw" | sed -E 's/[^a-zA-Z0-9]+/_/g' | sed -E 's/^_+|_+$//g' | tr 'A-Z' 'a-z')
  if [[ -z "$cleaned" ]]; then
    echo "all"
  else
    echo "$cleaned"
  fi
}

# Shorten a quantization type to a compact label.
# Examples:
#  float8_e4m3fn -> fp8
#  float4_e2m1fn -> fp4
#  int8 -> int8
#  bfloat16/bf16 -> bf16; float16/fp16 -> fp16; float32/fp32 -> fp32
vllm_qtype_short() {
  local qt="$1"
  if [[ -z "$qt" || "$qt" == "none" ]]; then echo ""; return; fi
  if [[ "$qt" =~ ^float([0-9]+)_ ]]; then echo "fp${BASH_REMATCH[1]}"; return; fi
  if [[ "$qt" =~ ^int([0-9]+) ]]; then echo "int${BASH_REMATCH[1]}"; return; fi
  case "$qt" in
    bfloat16|bf16) echo "bf16" ; return ;;
    float16|fp16|f16) echo "fp16" ; return ;;
    float32|fp32|f32) echo "fp32" ; return ;;
  esac
  echo "$qt"
}

# Compose a readable qtype name from weight/activation qtypes.
# If both are empty/none -> baseline. If equal -> single tag; else W_A.
vllm_qtype_name() {
  local wq="$1"; local aq="$2"
  local ws; ws=$(vllm_qtype_short "$wq")
  local as; as=$(vllm_qtype_short "$aq")
  if [[ -z "$ws" && -z "$as" ]]; then echo "baseline"; return; fi
  if [[ "$ws" == "$as" && -n "$ws" ]]; then echo "$ws"; return; fi
  if [[ -z "$ws" ]]; then echo "none_${as}"; return; fi
  if [[ -z "$as" ]]; then echo "${ws}_none"; return; fi
  echo "${ws}_${as}"
}
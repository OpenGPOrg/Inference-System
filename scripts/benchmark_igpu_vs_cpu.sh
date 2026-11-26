#!/usr/bin/env bash
# ==============================================================================
# OpenGP: Benchmark Script (Vulkan iGPU vs. Multi-Threaded CPU)
# Evaluates Prefill (Prompt Processing) and Decode (Token Generation) Speedup
# ==============================================================================

set -euo pipefail

# --- Default Configurations (Overridable via CLI flags or env vars) ---
MODEL_PATH="${MODEL_PATH:-models/qwen2-0_5b-instruct-q4_0.gguf}"
ITERATIONS="${ITERATIONS:-3}"
MAX_TOKENS="${MAX_TOKENS:-48}"
THREADS="${THREADS:-0}" # 0 = auto-detect hardware concurrency
BINARY="./build/opengp_infer"

# Standard test prompt (~140 tokens)
DEFAULT_PROMPT="Explain in detail how transformer neural networks process text from raw input strings through tokenization, embeddings, positional encoding, multi-head self-attention, key-value caching, feed-forward networks, residual connections, RMS normalization, and autoregressive token sampling. Please write a clear and concise technical summary covering each stage and how memory bandwidth affects prefill versus decode latency on integrated GPUs versus multi-core CPUs."
PROMPT="${PROMPT:-$DEFAULT_PROMPT}"

# --- Helper: Print Usage ---
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -n, --iterations <N>   Number of benchmark runs for each backend (default: 3)
  -t, --max-tokens <N>   Tokens to generate per iteration (default: 48)
  -m, --model <path>     Path to GGUF model file (default: $MODEL_PATH)
  -p, --prompt <text>    Custom prompt text
  -c, --threads <N>      CPU worker threads (default: 0 = auto-detect)
  -h, --help             Show this help message

Examples:
  $0 -n 5 -t 64
  $0 --iterations 3 --model models/qwen2-0_5b-instruct-q4_0.gguf
EOF
    exit 0
}

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--iterations) ITERATIONS="$2"; shift 2 ;;
        -t|--max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        -m|--model)      MODEL_PATH="$2"; shift 2 ;;
        -p|--prompt)     PROMPT="$2"; shift 2 ;;
        -c|--threads)    THREADS="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# --- Validations ---
if [[ ! -x "$BINARY" ]]; then
    echo "[-] Error: Executable $BINARY not found or not executable. Run 'make' first." >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[-] Error: Model file '$MODEL_PATH' not found." >&2
    exit 1
fi

echo "================================================================="
echo "        OPENGP BENCHMARK: VULKAN iGPU vs. CPU SPEEDUP           "
echo "================================================================="
echo "Model       : $MODEL_PATH"
echo "Iterations  : $ITERATIONS per backend"
echo "Max Tokens  : $MAX_TOKENS decode tokens per run"
echo "CPU Threads : $([[ "$THREADS" -eq 0 ]] && echo "Auto-detect" || echo "$THREADS")"
echo "================================================================="

# --- Helper: Run Single Benchmark Loop ---
# Arguments: $1 = Mode Label, $2... = Extra Flags
run_benchmark_mode() {
    local label="$1"
    shift
    local extra_flags=("$@")

    echo ""
    echo ">>> Running Benchmark: $label ($ITERATIONS iterations)..."

    local total_prefill=0
    local total_decode=0
    local count=0

    for ((i=1; i<=ITERATIONS; i++)); do
        printf "  [Run %d/%d] " "$i" "$ITERATIONS"

        # Capture output, extracting prefill and decode speeds
        local output
        output=$("$BINARY" \
            --model "$MODEL_PATH" \
            --prompt "$PROMPT" \
            --max-tokens "$MAX_TOKENS" \
            --temp 0 \
            "${extra_flags[@]}" 2>&1)

        # Parse metrics:
        # Expected lines:
        # [Stats] Prefill : 139 tokens in 1511.6 ms (92.0 tok/s)
        # [Stats] Decode  : 48 tokens in 3536.8 ms (13.57 tok/s) | Stop: ...
        local prefill_speed
        local decode_speed
        prefill_speed=$(echo "$output" | grep -E '^\[Stats\] Prefill' | sed -n -E 's/.*ms \(([0-9.]+) tok\/s\).*/\1/p')
        decode_speed=$(echo "$output" | grep -E '^\[Stats\] Decode' | sed -n -E 's/.*ms \(([0-9.]+) tok\/s\).*/\1/p')

        if [[ -n "$prefill_speed" && -n "$decode_speed" ]]; then
            printf "Prefill: %6.1f tok/s | Decode: %5.2f tok/s\n" "$prefill_speed" "$decode_speed"
            total_prefill=$(awk -v t="$total_prefill" -v s="$prefill_speed" 'BEGIN { printf "%.2f", t + s }')
            total_decode=$(awk -v t="$total_decode" -v s="$decode_speed" 'BEGIN { printf "%.2f", t + s }')
            count=$((count + 1))
        else
            echo "FAILED (Unexpected output)" >&2
            echo "$output" | grep -E '\[-\]|\[!\]|Error' || true
        fi
    done

    if [[ "$count" -gt 0 ]]; then
        local avg_prefill
        local avg_decode
        avg_prefill=$(awk -v t="$total_prefill" -v c="$count" 'BEGIN { printf "%.2f", t / c }')
        avg_decode=$(awk -v t="$total_decode" -v c="$count" 'BEGIN { printf "%.2f", t / c }')
        echo "  --> Average: Prefill: $avg_prefill tok/s | Decode: $avg_decode tok/s"
        # Export averages dynamically
        eval "${label}_PREFILL=$avg_prefill"
        eval "${label}_DECODE=$avg_decode"
    fi
}

# --- Optional Warm-up run for Vulkan Shader Cache ---
echo ""
echo "[*] Warming up Vulkan pipeline cache (1 quick pass)..."
"$BINARY" --model "$MODEL_PATH" --prompt "Warmup" --max-tokens 4 --temp 0 --device gpu --lm-head gpu >/dev/null 2>&1 || true

# 1. Benchmark Vulkan iGPU (explicit --device gpu --lm-head gpu to bypass AUTO iGPU speed gate)
run_benchmark_mode "VULKAN_IGPU" --device gpu --lm-head gpu

# 2. Benchmark Multi-Threaded CPU
CPU_FLAGS=(--device cpu --lm-head cpu)
if [[ "$THREADS" -gt 0 ]]; then
    CPU_FLAGS+=(--threads "$THREADS")
fi
run_benchmark_mode "CPU" "${CPU_FLAGS[@]}"

# --- Speedup Summary Report ---
echo ""
echo "================================================================="
echo "                       FINAL SUMMARY & SPEEDUP                   "
echo "================================================================="
printf "%-18s | %-16s | %-16s\n" "Backend" "Prefill (tok/s)" "Decode (tok/s)"
echo "-------------------|------------------|------------------"
printf "%-18s | %16.2f | %16.2f\n" "Vulkan iGPU" "$VULKAN_IGPU_PREFILL" "$VULKAN_IGPU_DECODE"
printf "%-18s | %16.2f | %16.2f\n" "CPU" "$CPU_PREFILL" "$CPU_DECODE"
echo "-------------------|------------------|------------------"

# Compute Speedup ratios
PREFILL_SPEEDUP=$(awk -v v="$VULKAN_IGPU_PREFILL" -v c="$CPU_PREFILL" 'BEGIN { if (c > 0) printf "%.2fx", v / c; else print "N/A" }')
DECODE_SPEEDUP=$(awk -v v="$VULKAN_IGPU_DECODE" -v c="$CPU_DECODE" 'BEGIN { if (c > 0) printf "%.2fx", v / c; else print "N/A" }')

printf "%-18s | %16s | %16s\n" "Speedup (iGPU/CPU)" "$PREFILL_SPEEDUP" "$DECODE_SPEEDUP"
echo "================================================================="
echo "Done!"

#!/bin/bash
# Qwen3-Omni Transformers Benchmark Evaluation Script with Profiling
# This script must be run from the vllm-omni root directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to vllm-omni root directory (4 levels up from script location)
VLLM_OMNI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$VLLM_OMNI_ROOT" || { echo "Error: Failed to navigate to vllm-omni directory"; exit 1; }

echo "Working directory: $(pwd)"
# Verify we're in the correct directory and run benchmark
if [[ ! -f "benchmarks/qwen3-omni/transformers/qwen3_omni_moe_transformers_profile.py" ]]; then
    echo "Error: Not in vllm-omni root directory. Please run from vllm-omni folder."
else
    cd benchmarks/qwen3-omni/transformers

    # Calculate path to thermal_headroom/data/raw/profiler
    # From benchmarks/qwen3-omni/transformers, go up 4 levels to thermal_headroom
    THERMAL_HEADROOM_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
    PROFILER_OUTPUT_DIR="$THERMAL_HEADROOM_ROOT/data/raw/profiler"
    
    # Create profiler output directory if it doesn't exist
    mkdir -p "$PROFILER_OUTPUT_DIR"
    
    echo "Profiler output directory: $PROFILER_OUTPUT_DIR"

    python qwen3_omni_moe_transformers_profile.py \
        --prompts_file ../../build_dataset/top1.txt \
        --num_prompts 1 \
        --enable-profiler \
        --profiler-output-dir "$PROFILER_OUTPUT_DIR"

    echo ""
    echo "Logs and outputs are saved to $(pwd)/benchmark_results:"
    echo "  - perf_stats.json    Aggregated/per-prompt TPS and latency (thinker/talker/code2wav/overall)"
    echo "  - results.json       Per-prompt outputs and audio paths"
    echo "  - audio/             Generated wav files"
    echo ""
    echo "Profiler traces are saved to: $PROFILER_OUTPUT_DIR"
    echo "  - trace_prompt_*.json    Chrome trace format files (viewable in Chrome://tracing or https://ui.perfetto.dev/)"
    echo ""
    echo "Key checks: overall_tps and *_tps_avg should be non-zero and stable; investigate 0/NaN or unusually low TPS/long-tail latency."
fi


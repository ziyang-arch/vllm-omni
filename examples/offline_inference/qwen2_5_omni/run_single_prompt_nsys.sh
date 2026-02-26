#!/bin/bash
# Qwen3-Omni Transformers Benchmark Evaluation Script with nsys Profiling
# This script must be run from the vllm-omni root directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Calculate path to thermal_headroom/data/raw/profiler
# From benchmarks/vllm-omni/examples/offline_inference/qwen2_5_omni, go up 4 levels to thermal_headroom
THERMAL_HEADROOM_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PROFILER_OUTPUT_DIR="$THERMAL_HEADROOM_ROOT/data/raw/profiler"

# Create profiler output directory if it doesn't exist
mkdir -p "$PROFILER_OUTPUT_DIR"

# Generate output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NSYS_OUTPUT="$PROFILER_OUTPUT_DIR/nsys_qwen2_5_omni_single_prompt_example_${TIMESTAMP}"

echo "Profiler output directory: $PROFILER_OUTPUT_DIR"
echo "nsys output file: ${NSYS_OUTPUT}.qdrep"

# GPU metrics are optional and often require admin enablement of perf counters.
# Enable with: NSYS_GPU_METRICS=1
NSYS_GPU_METRICS="${NSYS_GPU_METRICS:-0}"
GPU_METRICS_OPT=""
if [[ "$NSYS_GPU_METRICS" == "1" ]]; then
    GPU_METRICS_OPT="--gpu-metrics-device=all"
    echo "NSYS_GPU_METRICS=1 -> enabling GPU metrics (${GPU_METRICS_OPT})"
else
    echo "NSYS_GPU_METRICS=0 -> GPU metrics disabled (avoids ERR_NVGPUCTRPERM failures)"
fi

#--gpu-metrics-device=all


nsys profile \
        -o "$NSYS_OUTPUT" \
        --force-overwrite=true \
        --sample=cpu \
        --trace=cuda,osrt,nvtx \
        --cuda-memory-usage=true \
        ${GPU_METRICS_OPT} \
python end2end.py --output-wav output_audio \
                  --query-type use_mixed_modalities

#!/bin/bash
# Qwen3-Omni Transformers Benchmark Evaluation Script with nsys Profiling
# This script must be run from the vllm-omni root directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Calculate path to thermal_headroom/data/raw/profiler
# From benchmarks/vllm-omni/examples/offline_inference/qwen2_5_omni, go up 4 levels to thermal_headroom
THERMAL_HEADROOM_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PROFILER_OUTPUT_DIR="$THERMAL_HEADROOM_ROOT/data/raw/profiler"



VLLM_TORCH_PROFILER_DIR="$PROFILER_OUTPUT_DIR/torchprofile_qwen2_5_omni_single_prompt_example_$(date +%Y%m%d_%H%M%S)" python end2end.py --output-wav output_audio \
                  --query-type use_mixed_modalities

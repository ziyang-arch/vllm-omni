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
NCU_OUTPUT="$PROFILER_OUTPUT_DIR/ncu_roofline_qwen2_5_omni_single_prompt_example_${TIMESTAMP}"

  echo "Profiler output directory: $PROFILER_OUTPUT_DIR"
  echo "ncu output file: ${NCU_OUTPUT}.ncu-rep"
  echo "ncu output log: ${NCU_OUTPUT}.log"
  echo "shell output log: ${NCU_OUTPUT}_shell.log"

{
  echo "Profiler output directory: $PROFILER_OUTPUT_DIR"
  echo "ncu output file: ${NCU_OUTPUT}.ncu-rep"
  echo "ncu output log: ${NCU_OUTPUT}.log"
  echo "shell output log: ${NCU_OUTPUT}_shell.log"


# take out --set roofline     --csv \     --print-details all \     --page details \


sudo -E env "PATH=$PATH" ncu \
    --section SpeedOfLight_RooflineChart \
      --page details --print-details all --csv \
      --target-processes all --kernel-name-base demangled \
    --target-processes all \
    --kernel-name-base demangled \
    -f --export ${NCU_OUTPUT}_SoL_roofline.ncu-rep \
    python end2end.py --output-wav output_audio --query-type use_mixed_modalities


} &> ${NCU_OUTPUT}_shell.log 2>&1

# Clean up server and vllm processes
#sudo kill -9 $(pgrep -f "python end2end.py")
#kill -9 $(pgrep -f "python -m vllm.entrypoints.openai.api_server")
#sudo kill -9 $(pgrep -f "python")

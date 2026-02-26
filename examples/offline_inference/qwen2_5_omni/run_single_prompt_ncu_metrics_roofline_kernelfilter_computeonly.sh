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
NCU_OUTPUT="$PROFILER_OUTPUT_DIR/ncu_roofline_compute_only_qwen2_5_omni_single_prompt_example_${TIMESTAMP}"

  echo "Profiler output directory: $PROFILER_OUTPUT_DIR"
  echo "ncu output file: ${NCU_OUTPUT}.ncu-rep"
  echo "ncu output log: ${NCU_OUTPUT}.log"
  echo "shell output log: ${NCU_OUTPUT}_shell.log"

{
  echo "Profiler output directory: $PROFILER_OUTPUT_DIR"
  echo "ncu output file: ${NCU_OUTPUT}.ncu-rep"
  echo "ncu output log: ${NCU_OUTPUT}.log"
  echo "ncu output csv: ${NCU_OUTPUT}.csv"
  echo "shell output log: ${NCU_OUTPUT}_shell.log"


# take out --set roofline     --csv \     --print-details all \     --page details \


sudo -E env "PATH=$PATH" ncu \
    --metrics \
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_integer_pred_on.sum,\
sm__ops_path_tensor_src_fp16_dst_fp32.sum,\
sm__ops_path_tensor_src_bf16_dst_fp32.sum,\
sm__ops_path_tensor_src_tf32_dst_fp32.sum,\
sm__ops_path_tensor_src_fp16_dst_fp16.sum,\
sm__ops_path_tensor_src_fp64_dst_fp64.sum,\
dram__bytes.sum \
    --kernel-id "::regex:(ampere_.*_s16816gemm_.*|cutlass.*gemm.*|.*cutlass.*):1" \
    --target-processes all \
    --launch-count 10000 \
    --kernel-name-base demangled \
    --print-kernel-base demangled \
    -f --export ${NCU_OUTPUT}.ncu-rep \
    python end2end.py --output-wav output_audio --query-type use_mixed_modalities


} &> ${NCU_OUTPUT}_shell.log 2>&1

# Clean up server and vllm processes
#sudo kill -9 $(pgrep -f "python end2end.py")
#kill -9 $(pgrep -f "python -m vllm.entrypoints.openai.api_server")
#sudo kill -9 $(pgrep -f "python")

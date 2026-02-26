#!/bin/bash
# Qwen3-Omni Transformers Benchmark Evaluation Script with ncu Profiling
# Usage:
#   ./run_ncu_roofline.sh <launch_skip> <launch_count>

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <launch_skip> <launch_count>"
  exit 1
fi

LAUNCH_SKIP="$1"
LAUNCH_COUNT="$2"

echo "Using ncu --launch-skip=${LAUNCH_SKIP} --launch-count=${LAUNCH_COUNT}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THERMAL_HEADROOM_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PROFILER_OUTPUT_DIR="$THERMAL_HEADROOM_ROOT/data/raw/profiler"

mkdir -p "$PROFILER_OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Encode skip/count in filename
NCU_OUTPUT="$PROFILER_OUTPUT_DIR/ncu_roofline_qwen2_5_omni_single_prompt_example_skip${LAUNCH_SKIP}_cnt${LAUNCH_COUNT}_${TIMESTAMP}"

echo "Profiler output directory: $PROFILER_OUTPUT_DIR"
echo "ncu output file: ${NCU_OUTPUT}.ncu-rep"
echo "ncu output log: ${NCU_OUTPUT}.log"
echo "shell output log: ${NCU_OUTPUT}_shell.log"

{
  echo "Profiler output directory: $PROFILER_OUTPUT_DIR"
  echo "ncu output file: ${NCU_OUTPUT}.ncu-rep"
  echo "ncu output log: ${NCU_OUTPUT}.log"
  echo "shell output log: ${NCU_OUTPUT}_shell.log"
  echo "launch_skip: ${LAUNCH_SKIP}"
  echo "launch_count: ${LAUNCH_COUNT}"

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
    --kernel-id "::regex:^.*$:1" \
    --target-processes all \
    --kernel-name-base demangled \
    --print-kernel-base demangled \
    --launch-skip "${LAUNCH_SKIP}" \
    --launch-count "${LAUNCH_COUNT}" \
    -f --export "${NCU_OUTPUT}" \
    python end2end.py --output-wav output_audio --query-type use_mixed_modalities

} &> "${NCU_OUTPUT}_shell.log" 2>&1

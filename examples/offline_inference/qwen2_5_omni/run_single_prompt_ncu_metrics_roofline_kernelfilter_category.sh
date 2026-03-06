#!/bin/bash
# Qwen3-Omni Transformers Benchmark Evaluation Script with nsys Profiling
# This script must be run from the vllm-omni root directory

# Ensure a category argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <category>"
    echo "Available Categories:"
    echo "  gemm        - GEMM, Matmul, CUTLASS, XMMA, FlashAttention"
    echo "  elementwise - PyTorch/vLLM elementwise ops, activations (SiLU)"
    echo "  reduction   - Softmax, RMSNorm, LayerNorm, Reductions"
    echo "  attention   - vLLM PagedAttention reshape/cache"
    echo "  other       - Catch-all for remaining PyTorch/CUB utility kernels"
    exit 1
fi

CATEGORY=$1

# Map the category to the correct regex filter (capturing the first invocation ':1')
case $CATEGORY in
    gemm)
        REGEX_FILTER="::regex:.*(gemm|cutlass|xmma|fmha|flash).*:1"
        ;;
    elementwise)
        REGEX_FILTER="::regex:.*(elementwise|conv2d_grouped_direct|act_and_mul).*:1"
        ;;
    reduction)
        REGEX_FILTER="::regex:.*(softmax|norm|reduce).*:1"
        ;;
    attention)
        REGEX_FILTER="::regex:.*reshape_and_cache_flash.*:1"
        ;;
    other)
        REGEX_FILTER="::regex:.*(fill_reverse|masked_scatter|kaiser|rotary|sinc|gemvx|CatArray|pool|conv_depthwise|indexSelect|pad|write_indices|gather|DeviceCompact|RadixSort|DeviceScan|DeviceSelect|epilogue|dgrad2d|nchwToNhwc|nhwcToNchw|repetition_penalties).*:1"
        ;;
    *)
        echo "Error: Invalid category '$CATEGORY'"
        echo "Please use: gemm, elementwise, reduction, attention, or other."
        exit 1
        ;;
esac

echo "Selected Category: $CATEGORY"
echo "Using NCU Regex Filter: $REGEX_FILTER"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Calculate path to thermal_headroom/data/raw/profiler
# From benchmarks/vllm-omni/examples/offline_inference/qwen2_5_omni, go up 4 levels to thermal_headroom
THERMAL_HEADROOM_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PROFILER_OUTPUT_DIR="$THERMAL_HEADROOM_ROOT/data/raw/profiler"

# Create profiler output directory if it doesn't exist
mkdir -p "$PROFILER_OUTPUT_DIR"

# Generate output filename with timestamp AND the selected category
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NCU_OUTPUT="$PROFILER_OUTPUT_DIR/ncu_roofline_${CATEGORY}_qwen2_5_omni_${TIMESTAMP}"

{
  echo "Profiler output directory: $PROFILER_OUTPUT_DIR"
  echo "ncu output file: ${NCU_OUTPUT}.ncu-rep"
  echo "ncu output log: ${NCU_OUTPUT}.log"
  echo "ncu output csv: ${NCU_OUTPUT}.csv"
  echo "shell output log: ${NCU_OUTPUT}_shell.log"

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
    --kernel-id "$REGEX_FILTER" \
    --target-processes all \
    --replay-mode application \
    --launch-count 1000 \
    --kernel-name-base demangled \
    --print-kernel-base demangled \
    -f --export ${NCU_OUTPUT}.ncu-rep \
    python end2end.py --output-wav output_audio --query-type use_mixed_modalities

} &> ${NCU_OUTPUT}_shell.log 2>&1

# Clean up server and vllm processes
#sudo kill -9 $(pgrep -f "python end2end.py")
#kill -9 $(pgrep -f "python -m vllm.entrypoints.openai.api_server")
#sudo kill -9 $(pgrep -f "python")

echo "Profiling complete. Check ${NCU_OUTPUT}_shell.log for details."
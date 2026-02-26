#!/bin/bash
# Batch driver for Nsight Compute roofline profiling over kernel segments.
#
# It calls:
#   run_single_prompt_ncu_metrics_roofline_section.sh <launch_skip> <launch_count>
#
# Segments are sized at 100,000 kernels each, over a total of ~1,049,831 kernels.

set -euo pipefail

# Configurable parameters
SEGMENT_SIZE=100000          # kernels per segment
TOTAL_KERNELS=1049831        # total kernels in the program (approximate)
NUM_SEGMENTS=10              # how many segments to run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NCU_SCRIPT="${SCRIPT_DIR}/run_single_prompt_ncu_metrics_roofline_section.sh"

if [ ! -x "${RUN_NCU_SCRIPT}" ]; then
  echo "Error: ${RUN_NCU_SCRIPT} not found or not executable."
  exit 1
fi

echo "Total kernels: ${TOTAL_KERNELS}"
echo "Segments: ${NUM_SEGMENTS}, segment size: ${SEGMENT_SIZE}"
echo "Using script: ${RUN_NCU_SCRIPT}"

for (( i=0; i<NUM_SEGMENTS; i++ )); do
  START=$(( i * SEGMENT_SIZE ))
  REMAINING=$(( TOTAL_KERNELS - START ))
  if [ "${REMAINING}" -le 0 ]; then
    echo "No more kernels after segment ${i}, stopping."
    break
  fi

  COUNT=${SEGMENT_SIZE}
  if [ "${REMAINING}" -lt "${SEGMENT_SIZE}" ]; then
    COUNT=${REMAINING}
  fi

  echo "=== Segment ${i} : launch_skip=${START}, launch_count=${COUNT} ==="
  "${RUN_NCU_SCRIPT}" "${START}" "${COUNT}"
done

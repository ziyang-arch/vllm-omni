#!/usr/bin/env bash
# Run end2end.py with PyTorch profiler enabled

python end2end.py --output-dir output_audio \
                  --query-type use_audio \
                  --init-sleep-seconds 90 \
                  --enable-profiler \
                  --profiler-output-dir profiler_traces \
                  --enable-stats

# Notes:
# - --enable-profiler: Enables PyTorch profiler for detailed CPU/CUDA profiling
# - --profiler-output-dir: Directory to save profiler traces (Chrome trace format)
# - --enable-stats: Also enables vLLM-omni pipeline statistics logging
# - init-sleep-seconds works to avoid two vLLM stages initialized at the same time within a card.
# - Profiler traces can be viewed in Chrome://tracing or https://ui.perfetto.dev/


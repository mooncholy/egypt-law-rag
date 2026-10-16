#!/usr/bin/env bash
# sample_usage.sh <out.csv> <seconds>: GPU use and each container's CPU and memory, every 5 s.
set -euo pipefail
out=$1; end=$((SECONDS + $2))
echo "t,gpu_util_pct,gpu_mem_mib,container,cpu_pct,mem" > "$out"
while [ $SECONDS -lt $end ]; do
  t=$(date +%s)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')
  docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' raglaw-api-1 raglaw-models-1 raglaw-ollama-1 2>/dev/null \
    | while IFS= read -r line; do echo "$t,$gpu,$line" >> "$out"; done
  sleep 5
done

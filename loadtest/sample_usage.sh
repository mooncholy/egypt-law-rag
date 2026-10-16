#!/usr/bin/env bash
# sample_usage.sh <out.csv> <seconds>: GPU use and each container's CPU and memory, every 5 s.
set -euo pipefail
out=$1; end=$((SECONDS + $2))
echo "t,gpu_util_pct,gpu_mem_mib,container,cpu_pct,mem" > "$out"
while [ $SECONDS -lt $end ]; do
  t=$(date +%s)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')
  # Whichever generator runs (Ollama or vLLM), beside api and models.
  running=$(docker ps --format '{{.Names}}' | grep -E '^raglaw-(api|models|ollama|vllm)-1$' || true)
  # shellcheck disable=SC2086 # one container name per word
  docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' $running 2>/dev/null \
    | while IFS= read -r line; do echo "$t,$gpu,$line" >> "$out"; done
  sleep 5
done

#!/usr/bin/env bash
# Train every config × seed sequentially. Usage: scripts/run_grid.sh "cfgA cfgB" "42 1 2"
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MLFLOW_DISABLE_AGENT_HINT=1 HF_HUB_OFFLINE=1
mkdir -p data/logs
for cfg in $1; do
  for seed in $2; do
    log="data/logs/${cfg}_seed${seed}.log"
    python -m ml.training.train --config "ml/configs/${cfg}.yaml" --seed "$seed" > "$log" 2>&1
    echo "$(date +%H:%M) ${cfg} seed${seed} exit=$? $(grep -o 'early stop.*' "$log") $(tr -d '\n ' < "$log" | grep -o '"results":.*' | head -c 400)"
  done
done

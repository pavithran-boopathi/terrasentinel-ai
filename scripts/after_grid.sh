#!/usr/bin/env bash
# Wait for the Phase 4 grid to finish, then run the remaining GPU jobs sequentially.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MLFLOW_DISABLE_AGENT_HINT=1 HF_HUB_OFFLINE=1
while pgrep -f "scripts/run_grid.sh" >/dev/null; do sleep 20; done
echo "$(date +%H:%M) grid finished"
python -m ml.training.train_spacenet8 --epochs 40 --patience 10 > data/logs/sn8_train.log 2>&1
echo "$(date +%H:%M) sn8 exit=$?"
python -m ml.training.loeo --arch unet_resnet34 --epochs 35 > data/logs/loeo.log 2>&1
echo "$(date +%H:%M) loeo exit=$?"
python scripts/benchmark_latency.py > data/logs/latency.log 2>&1
echo "$(date +%H:%M) latency exit=$?"

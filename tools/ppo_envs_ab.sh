#!/bin/bash
# Does learning per sample survive a larger number of parallel envs? Same 12x12 starting weights, 4 M steps each, minibatch size fixed at
# 2048 samples (so gradient steps per sample are unchanged). Compare the training score over the last epochs and the greedy test fill.
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONPATH=python
PY=./.venv-cuda/Scripts/python.exe
for N in 256 1024 2048; do
  $PY python/train_ppo.py --name ab_envs$N --device cuda --size 12 --obs win6 --mask --gamma 0.995 --win-reward 10 \
     --init-from models/_runs/init_12x12_ext.pt --total-steps 4000000 --epoch-steps 500000 --checkpoint-every 8 \
     --envs $N --minibatches $((N / 32)) --seed 7 > results/ab_envs$N.log 2>&1
done

#!/bin/bash
# Throughput levers for the 12x12 PPO job on the GTX 1650 Ti. Run with the GPU otherwise idle (writes results/bench/ppo_levers.json).
# Minibatch size is kept at 2048 samples whatever the number of envs, so the number of gradient steps per sample stays the same.
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONPATH=python
PY=./.venv-cuda/Scripts/python.exe
P="$PY python/profile_ppo.py"
$P --envs 256 --split --tag baseline_lazy_split
$P --envs 256 --no-lazy --tag no_lazy_next_obs
$P --envs 256 --no-cudnn-benchmark --tag no_cudnn_benchmark
$P --envs 512  --minibatches 16 --tag envs512
$P --envs 1024 --minibatches 32 --tag envs1024
$P --envs 2048 --minibatches 64 --tag envs2048
$P --envs 256 --amp --tag amp
$P --envs 256 --channels-last --tag channels_last
$P --envs 256 --amp --channels-last --tag amp_channels_last

# Pause / resume for long training runs (power cut, laptop sleep, reboot).
#   * Stop cleanly:  create the file  results/STOP  (all jobs)  or  results/STOP_<run name>  (one job). A running job checks for it
#     once per iteration, writes models/_runs/<run>_state.pt and exits with code 0. Delete the file before resuming.
#   * Resume:        rerun the same command with  --resume  (see RESUME.md).
# The state file holds the network(s), the optimiser, the step / update counters, the per-epoch metrics and checkpoint evaluations,
# the best-validation score and the RNG states, and is also refreshed at every logging epoch. NOT saved: the environments' games in
# progress (they restart from the start position) and, for DQN, the replay buffer (it refills during the learn-start warm-up).
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def state_path(name):
    return ROOT / "models" / "_runs" / f"{name}_state.pt"


def stop_requested(name):
    return (ROOT / "results" / "STOP").exists() or (ROOT / "results" / f"STOP_{name}").exists()


def save_state(name, payload):
    payload = dict(payload, rng={"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
    p = state_path(name)
    tmp = p.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, p)   # atomic: a crash while saving never leaves a half-written state


def load_state(name, device):
    p = state_path(name)
    if not p.exists():
        return None
    st = torch.load(p, map_location=device, weights_only=False)
    torch.set_rng_state(st["rng"]["cpu"].cpu())
    if st["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["rng"]["cuda"]])
    return st

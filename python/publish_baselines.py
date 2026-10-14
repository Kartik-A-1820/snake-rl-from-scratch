# (Re)publish the existing agents into the model registry (models/<algo_id>/): the tabular Q table and the DQN
# checkpoints chosen as the best of tryouts 01 and 02. Safe to re-run. New tryouts publish themselves
# (train_dqn.py --publish ...), this script only exists to migrate what was trained before the registry.
#   python python/publish_baselines.py
import json
from pathlib import Path

from snake_rl import registry
from snake_rl.export import export_checkpoint

ROOT = registry.ROOT


def tabular():
    src = ROOT / "models" / "_runs" / "tabular_q_12x12_seed1.json"
    model = json.loads(src.read_text(encoding="utf-8"))
    test = registry.eval_tabular(model, 12)
    meta = {"name": "Tabular Q-learning (11 features)", "tryout": 1, "family": "tabular",
            "description": "Q-table over the 2048 states of 11 binary features (danger ahead/right/left, heading, food direction). "
                           "No body information beyond the three adjacent cells.",
            "obs": "basic", "size": 12, "size_locked": False, "params": len(model["q"]) * 3, "onnx_cpu_latency_us": None,
            "select_score": registry.eval_tabular(model, 12, registry.VAL_SEEDS)["mean"], "test": {k: test[k] for k in ("n", "mean", "std", "median", "p10", "p90", "max", "wins", "causes")},
            "train": None, "notes": "../notes/01_tabular_q_learning.html", "source_ckpt": src.name}
    registry.publish("01_tabular_q", model, meta)
    print(f"01_tabular_q: test mean {test['mean']:.2f} max {test['max']}")


def main():
    tabular()
    export_checkpoint(ROOT / "models" / "_runs" / "diag_A_basic_origHP_best.pt", "02_dqn_mlp", "DQN (MLP, 11 features)", 2,
                      "../notes/02_dqn.html",
                      "Same DQN as the CNN agent but fed the 11 hand-made features (the ablation in tryout 02).")
    export_checkpoint(ROOT / "models" / "_runs" / "dqn_cnn_12x12_long_best.pt", "02_dqn_cnn", "DQN (CNN, grid)", 2,
                      "../notes/02_dqn.html",
                      "Double DQN, 3-step returns, CNN on the 4x14x14 board image. 5 M env steps on CUDA.")


if __name__ == "__main__":
    main()

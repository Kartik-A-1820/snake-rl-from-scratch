# Rebuilds results/phase4_ppo/<run>/summary.json for a PPO run that was stopped early (the trainer writes the summary only at
# the very end). Uses the run's config.json, its per-epoch metrics.jsonl (which holds every checkpoint evaluation) and the
# best checkpoint models/_runs/<run>_best.pt; also records up to 3 winning games of that checkpoint as replays.
#   python python/rebuild_summary.py t05_6x6_win5_s0 [--search 120]
import argparse
import json
from pathlib import Path

import torch

from snake_rl.core import SnakeGame
from snake_rl.features import basic_features, win_obs
from snake_rl.nets import ActorCritic

ROOT = Path(__file__).resolve().parents[1]
TEST_SEEDS = list(range(100000, 100500))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--search", type=int, default=120, help="test seeds to try when looking for winning games to record")
    a = ap.parse_args()
    d = ROOT / "results" / "phase4_ppo" / a.run
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    metrics = [json.loads(line) for line in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cps = [m["checkpoint"] for m in metrics if "checkpoint" in m]
    ck = torch.load(ROOT / "models" / "_runs" / f"{a.run}_best.pt", map_location="cpu")
    c = ck["config"]
    summary = {"tryout": "04_ppo", "name": a.run, "config": cfg, "epochs": metrics, "checkpoints": cps,
               "best": {"epoch": ck["epoch"], "val_mean": ck["val_mean"], "test": ck["test"]}, "first_win": None, "stopped_early": True}
    if c["obs"] in ("win5", "win6") and ck["test"].get("wins"):
        net = ActorCritic(c["net_size"], in_ch=c["in_ch"])
        net.load_state_dict(ck["state_dict"])
        net.eval()
        size, reps = c["size"], []
        with torch.no_grad():
            for seed in TEST_SEEDS[:a.search]:
                g, acts = SnakeGame(size, size, seed=seed), []
                while not g.done:
                    q = net(torch.as_tensor(win_obs(g, c["obs"] == "win6")).unsqueeze(0))[0].tolist()
                    bad = basic_features(g)[:3] if c.get("mask") else [0, 0, 0]
                    bad = [0, 0, 0] if all(bad) else bad
                    act = max((i for i in range(3) if not bad[i]), key=lambda i: (q[i], -i))
                    acts.append(act)
                    g.step(act)
                if g.won:
                    reps.append({"seed": seed, "width": size, "height": size, "actions": "".join(map(str, acts)), "score": g.score, "cause": g.cause})
                    if len(reps) == 3:
                        break
        first = next((cp for cp in cps if cp["test"]["wins"]), None)
        summary["first_win"] = {"epoch": first["epoch"] if first else ck["epoch"], "env_steps": first["env_steps"] if first else None, "replays": reps}
    (d / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {d / 'summary.json'}: {len(cps)} checkpoints, best epoch {ck['epoch']} (test wins {ck['test'].get('wins')}), "
          f"{len(summary['first_win']['replays']) if summary['first_win'] else 0} win replays")


if __name__ == "__main__":
    main()

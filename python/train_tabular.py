# Tryout 01 in Python: tabular Q-learning on the 11 basic features.
# Same algorithm, same engine rules and same RNG as tools/train_tabular.js, so with the same seed it must
# reproduce the JS numbers exactly (seed 1, 12x12: 2,173,663 env steps, final eval mean 20.078).
# That makes it a good end-to-end parity check as well as the hands-on version of tryout 01.
#
#   python python/train_tabular.py --size 12 --epochs 200
#
# No PyTorch: a 2048x3 table needs no GPU (see notes/01_tabular_q_learning).
import argparse
import json
import math
import time
from pathlib import Path

from snake_rl.core import Rng, SnakeGame
from snake_rl.features import basic_index

ROOT = Path(__file__).resolve().parents[1]


def argmax(row):
    # first maximum, like the JS engine (ties → straight); numpy.argmax would also do, but this is a 3-list
    b = 0
    for i in (1, 2):
        if row[i] > row[b]:
            b = i
    return b


def evaluate(q, size, seeds):
    # greedy (ε = 0) on held-out seeds; returns the same summary fields as the JS evaluate()
    scores, causes = [], {}
    for seed in seeds:
        g = SnakeGame(size, size, seed=seed)
        while not g.done:
            g.step(argmax(q[basic_index(g)]))
        scores.append(g.score)
        causes[g.cause] = causes.get(g.cause, 0) + 1
    s = sorted(scores)
    pct = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    mid = s[int(len(s) * 0.25): math.ceil(len(s) * 0.75)]  # IQM: mean of the middle 50%
    return {"n": len(s), "mean": sum(s) / len(s), "median": pct(0.5), "p10": pct(0.1), "p90": pct(0.9),
            "max": s[-1], "iqm": sum(mid) / len(mid), "wins": causes.get("win", 0), "causes": causes}


def train(size=12, epochs=200, episodes_per_epoch=100, checkpoint_every=25, alpha=0.1, gamma=0.95,
          eps_start=1.0, eps_end=0.01, eps_decay_frac=0.5, seed=1, eval_episodes=500, eval_seed0=100000):
    total = epochs * episodes_per_epoch
    q = [[0.0, 0.0, 0.0] for _ in range(2048)]  # Q(s, a): 11 feature bits → row, 3 relative actions
    visits = [0] * 2048
    rng = Rng(seed ^ 0x9E3779B9)  # exploration RNG, separate from the food RNG inside the game
    g = SnakeGame(size, size, seed=seed)
    eval_seeds = list(range(eval_seed0, eval_seed0 + eval_episodes))
    window = {"scores": [], "wins": 0, "causes": {}}
    checkpoints, steps, t0 = [], 0, time.time()

    def checkpoint(epoch, eps):
        ev = evaluate(q, size, eval_seeds)
        sc = window["scores"]
        prev = checkpoints[-1]["eval"]["mean"] if checkpoints else None
        cp = {"epoch": epoch, "episodes": epoch * episodes_per_epoch, "env_steps": steps, "eps": round(eps, 3),
              "states_visited": sum(v > 0 for v in visits),
              "train": {"mean": sum(sc) / len(sc), "max": max(sc), "wins": window["wins"], "causes": window["causes"]} if sc else None,
              "eval": ev, "delta_eval_mean": None if prev is None else ev["mean"] - prev,
              "best_fill_pct": round(100 * (ev["max"] + 3) / (size * size), 1), "seconds": round(time.time() - t0, 1)}
        checkpoints.append(cp)
        tr = cp["train"] or {"mean": 0.0, "max": 0}
        delta = "" if prev is None else f"{ev['mean'] - prev:+.2f}"
        print(f"epoch {epoch:4d} | eps {eps:.3f} | train {tr['mean']:6.2f} (max {tr['max']:3d}) | "
              f"EVAL mean {ev['mean']:6.2f} median {ev['median']:3d} max {ev['max']:3d} wins {ev['wins']:3d} | "
              f"Δ {delta:>6} | {cp['seconds']}s", flush=True)
        window.update(scores=[], wins=0, causes={})

    checkpoint(0, eps_start)
    for ep in range(total):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (total * eps_decay_frac))
        g.reset()  # continue the food RNG stream: every episode is different, but the run is reproducible
        s = basic_index(g)
        while not g.done:
            # ε-greedy behaviour policy (draw order matches the JS trainer exactly)
            a = rng.int(3) if rng.next_u32() / 4294967296 < eps else argmax(q[s])
            r, terminated, truncated, _ = g.step(a)
            s2 = None if terminated else basic_index(g)
            # TD target: no bootstrap past a real terminal; DO bootstrap through a timeout (truncation)
            target = r + (0.0 if terminated else gamma * max(q[s2]))
            q[s][a] += alpha * (target - q[s][a])  # Q-learning update: move toward the target by α·δ
            visits[s] += 1
            s = s2
            steps += 1
        window["scores"].append(g.score)
        window["causes"][g.cause] = window["causes"].get(g.cause, 0) + 1
        window["wins"] += int(g.won)
        epoch = (ep + 1) / episodes_per_epoch
        if epoch.is_integer() and int(epoch) % checkpoint_every == 0:
            checkpoint(int(epoch), eps)

    model = {"type": "tabular", "features": "basic", "size": size,
             "q": {str(i): [round(v, 4) for v in row] for i, row in enumerate(q) if visits[i]}}
    return {"config": {"size": size, "epochs": epochs, "episodes_per_epoch": episodes_per_epoch, "alpha": alpha,
                       "gamma": gamma, "eps_start": eps_start, "eps_end": eps_end, "eps_decay_frac": eps_decay_frac,
                       "seed": seed, "eval_episodes": eval_episodes, "eval_seed0": eval_seed0},
            "win_score": size * size - 3, "checkpoints": checkpoints, "model": model}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--publish", action="store_true", help="publish to models/01_tabular_q/ if better on validation seeds")
    a = ap.parse_args()
    res = train(size=a.size, epochs=a.epochs, seed=a.seed, alpha=a.alpha, gamma=a.gamma)
    tag = f"{a.size}x{a.size}_seed{a.seed}_py"
    out = ROOT / "results" / "phase1_tabular"
    out.mkdir(parents=True, exist_ok=True)
    model = res.pop("model")
    (out / f"checkpoints_{tag}.json").write_text(json.dumps(res, indent=1))
    (ROOT / "models" / "_runs").mkdir(parents=True, exist_ok=True)
    (ROOT / "models" / "_runs" / f"tabular_q_{tag}.json").write_text(json.dumps(model))
    print(f"\nsaved results/phase1_tabular/checkpoints_{tag}.json and models/_runs/tabular_q_{tag}.json")
    if a.publish:
        from snake_rl import registry
        val = registry.eval_tabular(model, a.size, registry.VAL_SEEDS)["mean"]
        if registry.should_replace("01_tabular_q", val):
            test = registry.eval_tabular(model, a.size)
            meta = {"name": "Tabular Q-learning (11 features)", "tryout": 1, "family": "tabular",
                    "description": "Q-table over the 2048 states of 11 binary features. No body information beyond the three adjacent cells.",
                    "obs": "basic", "size": a.size, "size_locked": False, "params": len(model["q"]) * 3, "onnx_cpu_latency_us": None,
                    "select_score": val, "val_mean": val,
                    "test": {k: test[k] for k in ("n", "mean", "std", "median", "p10", "p90", "max", "wins", "causes")},
                    "train": None, "notes": "../notes/01_tabular_q_learning.html", "source_ckpt": f"tabular_q_{tag}.json"}
            registry.publish("01_tabular_q", model, meta)
            print(f"published models/01_tabular_q (val {val:.2f}, test {test['mean']:.2f})")
        else:
            print("registered tabular model has a higher validation mean; not replaced")

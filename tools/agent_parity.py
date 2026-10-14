# Checks that every registered agent plays identically in Python (ONNX Runtime + the scalar engine) and in the browser game.
# Step 1 (this script): play a few held-out seeds per agent in Python and write results/agent_parity.json.
# Step 2: open game/index.html (file://) and run the in-page comparison (see tools/agent_parity.md), or use the Chrome DevTools
# check described in the README. A mismatch means the exported JSON, the JS observation or the mask rule drifted from Python.
#   python tools/agent_parity.py [--seeds 100000 100001 100002]
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from snake_rl.core import SnakeGame, greedy_action  # noqa: E402
from snake_rl.features import basic_features, basic_index, grid_obs, trap_danger, win_obs  # noqa: E402
from snake_rl.hamilton import ham_action  # noqa: E402


def make_policy(meta, d):
    if meta["family"] == "tabular":
        q = json.loads((d / "best.json").read_text(encoding="utf-8"))["q"]
        return lambda g: max(range(3), key=lambda i: (q.get(str(basic_index(g)), [0, 0, 0])[i], -i))
    if meta["family"] == "planning":
        return ham_action
    import onnxruntime as ort
    sess = ort.InferenceSession(str(d / "best.onnx"), providers=["CPUExecutionProvider"])

    def obs(g):
        if meta["obs"] == "grid":
            return grid_obs(g)
        if meta["obs"] in ("win5", "win6"):
            return win_obs(g, meta["obs"] == "win6")
        return np.asarray(basic_features(g), dtype=np.float32)

    def act(g):
        q = sess.run(None, {"obs": obs(g)[None].astype(np.float32)})[0][0]
        bad = [False] * 3
        if meta.get("mask") == "trap":
            bad = trap_danger(g)
        elif meta.get("mask"):
            bad = [bool(b) for b in basic_features(g)[:3]]
        ok = [not b for b in bad]
        if not any(ok):
            ok = [True] * 3
        return max((i for i in range(3) if ok[i]), key=lambda i: (q[i], -i))
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[100000, 100001, 100002])
    a = ap.parse_args()
    out = {}
    for mp in sorted((ROOT / "models").glob("*/meta.json")):
        meta = json.loads(mp.read_text(encoding="utf-8"))
        pol = make_policy(meta, mp.parent)
        res = []
        for seed in a.seeds:
            g = SnakeGame(meta["size"], meta["size"], seed=seed)
            while not g.done:
                g.step(pol(g))
            res.append([g.score, g.cause, g.steps])
        out[meta["id"]] = {"size": meta["size"], "seeds": a.seeds, "results": res}
        print(meta["id"], [r[0] for r in res])
    (ROOT / "results" / "agent_parity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote results/agent_parity.json")


if __name__ == "__main__":
    main()

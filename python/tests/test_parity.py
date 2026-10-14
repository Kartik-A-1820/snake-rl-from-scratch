# The JS engine in game/index.html defines the rules. These tests prove the Python scalar engine and the
# batched torch env reproduce it exactly: same RNG draws, same food cells, same deaths, same observations.
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from snake_rl.core import Rng, SnakeGame, greedy_action  # noqa: E402
from snake_rl.features import basic_features, grid_obs, obs_digest  # noqa: E402
from snake_rl.vec_env import CAUSES, TorchSnakeVecEnv, mulberry32  # noqa: E402

TRACES = ROOT / "python" / "tests" / "js_traces.json"
# First outputs of mulberry32(42), recorded from the JS engine in a browser.
RNG42 = [2581720956, 1925393290, 3661312704]


def load_traces():
    if not TRACES.exists():
        node = shutil.which("node")
        if not node:
            pytest.skip("node not found: install Node.js or run `node tools/make_trace.js` once")
        subprocess.run([node, str(ROOT / "tools" / "make_trace.js"), str(TRACES)], check=True)
    return json.loads(TRACES.read_text())


def test_rng_matches_js():
    r = Rng(42)
    assert [r.next_u32() for _ in range(3)] == RNG42
    s = torch.tensor([42], dtype=torch.long)
    got = []
    for _ in range(3):
        s, u = mulberry32(s)
        got.append(int(u))
    assert got == RNG42


def test_rng_full_sequence_matches_js():
    tr = load_traces()
    r = Rng(42)
    assert [r.next_u32() for _ in range(len(tr["rng42"]))] == tr["rng42"]


def test_scalar_engine_matches_js_traces():
    tr = load_traces()
    for case in tr["cases"]:
        g = SnakeGame(case["size"], case["size"], seed=case["seed"])
        for t, st in enumerate(case["steps"]):
            where = f"size={case['size']} seed={case['seed']} t={t}"
            rew, term, trunc, ev = g.step(st["a"])
            assert (rew, term, trunc, ev) == (st["rew"], st["term"], st["trunc"], st["ev"]), where
            assert list(g.body[0]) == st["head"] and g.dir == st["dir"], where
            assert (list(g.food) if g.food else None) == st["food"], where
            assert (g.score, len(g.body)) == (st["score"], st["len"]), where
            bits = sum(b << i for i, b in enumerate(basic_features(g)))
            assert bits == st["feat"], where
            assert obs_digest(grid_obs(g)) == pytest.approx(st["obs"], rel=1e-6, abs=1e-6), where
            if g.done:
                g.reset()


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("size", [4, 6, 12])
def test_vec_env_matches_scalar(device, size):
    n, steps = 48, 1000  # ~1-2 min per case: the per-env Python comparison dominates, not the env
    seeds = list(range(1000, 1000 + n))
    games = [SnakeGame(size, size, seed=s) for s in seeds]
    env = TorchSnakeVecEnv(n, size, seeds=seeds, device=device)
    rnd = random.Random(0)
    for t in range(steps):
        acts = [greedy_action(g) if rnd.random() < 0.8 else rnd.randrange(3) for g in games]
        _, rew, term, trunc, info = env.step(torch.tensor(acts))
        nxt = info["next_obs"].cpu().numpy()
        for i, g in enumerate(games):
            where = f"{device} size={size} env={i} t={t}"
            r, te, tu, _ = g.step(acts[i])
            assert float(rew[i]) == r and bool(term[i]) == te and bool(trunc[i]) == tu, where
            if te or tu:
                assert CAUSES[int(info["cause"][i])] == g.cause, where
                assert int(info["score"][i]) == g.score, where
            if not g.won:  # after a win there is no food, so the food channel is meaningless
                np.testing.assert_allclose(nxt[i], grid_obs(g), atol=1e-7, err_msg=where)
            if g.done:
                g.reset()
            assert env.body(i) == list(g.body), where
            assert [int(v) for v in env.food[i]] == list(g.food), where
            assert int(env.dir[i]) == g.dir and int(env.score[i]) == g.score, where
        basic = env._basic().cpu().numpy()
        for i, g in enumerate(games):
            assert basic[i].tolist() == basic_features(g), f"basic features env={i} t={t}"

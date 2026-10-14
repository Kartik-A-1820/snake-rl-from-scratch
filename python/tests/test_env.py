import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from snake_rl.env import SnakeEnv  # noqa: E402


def test_gymnasium_api():
    check_env(SnakeEnv(obs="grid"), skip_render_check=True)
    check_env(SnakeEnv(obs="basic"), skip_render_check=True)


def test_seeded_reset_is_reproducible():
    e = SnakeEnv()
    a, _ = e.reset(seed=7)
    e.step(0)
    b, _ = e.reset(seed=7)
    np.testing.assert_array_equal(a, b)


def test_truncation_is_not_termination():
    e = SnakeEnv()
    e.reset(seed=0)
    while True:
        _, r, term, trunc, info = e.step(1)  # circling in a 2x2 square never dies
        if term or trunc:
            break
    assert trunc and not term and info["cause"] == "timeout"


def test_vec_env_shaping_matches_scalar_env():
    # Same seed + same random actions: shaped rewards of the torch vec env must equal the scalar SnakeEnv's.
    import torch
    from snake_rl.vec_env import TorchSnakeVecEnv

    rng = np.random.default_rng(0)
    for mode in ("none", "naive", "potential"):
        for pen in (0.0, 2.0):
            seed = 3
            sc = SnakeEnv(obs="basic", shaping=None if mode == "none" else mode, shape_k=0.7, timeout_penalty=pen)
            sc.reset(seed=seed)
            ve = TorchSnakeVecEnv(1, 12, seeds=[seed], obs="basic", shaping=mode, shape_k=0.7, timeout_penalty=pen)
            for _ in range(600):
                a = int(rng.integers(3))
                _, r_s, term_s, trunc_s, _ = sc.step(a)
                _, r_v, term_v, trunc_v, _ = ve.step(torch.tensor([a]))
                assert abs(r_s - float(r_v[0])) < 1e-5, (mode, pen, r_s, float(r_v[0]))
                assert term_s == bool(term_v[0]) and trunc_s == bool(trunc_v[0])
                if term_s or trunc_s:
                    sc.reset()  # no seed: continue the engine stream, like the vec env's auto-reset


def test_potential_shaping_telescopes_on_a_trajectory():
    # Sum of gamma^t * F over an episode = -phi(s_0) + gamma^T * phi(s_T): the policy-invariance identity.
    import torch
    from snake_rl.vec_env import TorchSnakeVecEnv

    g, k, S = 0.99, 1.0, 12
    plain = TorchSnakeVecEnv(1, S, seeds=[5], obs="basic", shaping="none")
    pot = TorchSnakeVecEnv(1, S, seeds=[5], obs="basic", shaping="potential", shape_k=k, gamma=g)
    rng = np.random.default_rng(1)
    phi = lambda e: -k * float((e.head - e.food).abs().sum()) / (2 * S)
    phi_start, disc, F_sum, t = phi(pot), 1.0, 0.0, 0
    while True:
        a = torch.tensor([int(rng.integers(3))])
        _, r0, t0, tr0, _ = plain.step(a)
        _, r1, t1, tr1, _ = pot.step(a)
        F_sum += disc * (float(r1[0]) - float(r0[0]))
        disc *= g
        t += 1
        if t1[0] or tr1[0]:
            break
    # death => phi(s_T) = 0, so the discounted shaping sum is exactly -phi(s_0) (when the episode ended by death/win)
    if bool(t1[0]):
        assert abs(F_sum - (-phi_start)) < 1e-4, (F_sum, -phi_start)

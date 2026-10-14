# Renders GIFs of the registered agents playing the same seed side by side (README hero images).
#   python tools/make_gif.py --out docs/agents.gif --agents 01_tabular_q 02_dqn_cnn 04_ppo --greedy
#   python tools/make_gif.py --out docs/wins.gif --agents 05_ppo_win_6x6 05_ppo_win_8x8 05_ppo_win_10x10 05_hamiltonian --seed-search
# Each agent plays on its own board size, with its own mask rule. Long games are shown at a coarser step so that every game fits the
# frame budget. Uses the Python mirror of the game engine and each agent's models/<id>/ files (ONNX Runtime for the networks).
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from snake_rl.core import SnakeGame, greedy_action  # noqa: E402
from snake_rl.features import basic_features, basic_index, grid_obs, trap_danger, win_obs  # noqa: E402
from snake_rl.hamilton import ham_action  # noqa: E402

LCD, DARK, MID, BG = (155, 188, 15), (15, 56, 15), (48, 98, 48), (29, 31, 27)


def load_agent(algo_id):
    d = ROOT / "models" / algo_id
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta["family"] == "planning":
        return meta, ham_action
    if meta["family"] == "tabular":
        q = json.loads((d / "best.json").read_text(encoding="utf-8"))["q"]
        return meta, lambda g: max(range(3), key=lambda i: (q.get(str(basic_index(g)), [0, 0, 0])[i], -i))
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
    return meta, act


def play(size, seed, act, cap=20000):
    g = SnakeGame(size, size, seed=seed)
    states = [(list(g.body), g.food, g.score, None)]
    while not g.done and len(states) < cap:
        g.step(act(g))
        states.append((list(g.body), g.food, g.score, g.cause if g.done else None))
    return states, g


def draw_panel(state, size, title, cell, font, H):
    body, food, score, cause = state
    W = max(size * cell + 16, 132)
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((8, 4), title, fill=(232, 234, 223), font=font)
    d.text((8, 20), f"score {score}" + (f"  x {cause}" if cause else ""), fill=LCD, font=font)
    top = 40
    d.rectangle([4, top - 2, 8 + size * cell + 4, top + size * cell + 4], fill=LCD)
    if food:
        d.rectangle([8 + food[0] * cell + cell // 4, top + 2 + food[1] * cell + cell // 4, 8 + food[0] * cell + 3 * cell // 4, top + 2 + food[1] * cell + 3 * cell // 4], fill=DARK)
    for i, (x, y) in enumerate(body):
        d.rectangle([8 + x * cell + 1, top + 2 + y * cell + 1, 8 + (x + 1) * cell - 2, top + 2 + (y + 1) * cell - 2], fill=DARK if i == 0 else MID)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--seed-search", action="store_true", help="use the first held-out seed that every learned agent wins")
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--greedy", action="store_true", help="add the greedy heuristic (12x12) as the first panel")
    ap.add_argument("--out", default=str(ROOT / "docs" / "agents.gif"))
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--cell", type=int, default=14)
    a = ap.parse_args()
    agents = [load_agent(i) for i in a.agents]
    def label(m):   # short titles: the panels of small boards are narrow
        if m["family"] == "planning":
            return "Hamiltonian (not RL)"
        if m["id"].startswith("05_ppo_win"):
            return f"PPO {m['size']}x{m['size']} win agent"
        return m["name"]
    players = ([("greedy heuristic", 12, greedy_action)] if a.greedy else []) + [(label(m), m["size"], act) for m, act in agents]
    seed = a.seed
    runs = None
    for attempt in range(40 if a.seed_search else 1):
        runs = [play(size, seed, act) for _, size, act in players]
        if not a.seed_search or all(g.won for (_, g), (n, _, _) in zip(runs, players) if "Hamiltonian" not in n):
            break
        seed += 1
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    H = max(s * a.cell for _, s, _ in players) + 52
    strides = [max(1, -(-(len(st) - 1) // a.frames)) for st, _ in runs]
    n_frames = max(-(-(len(st) - 1) // s) for (st, _), s in zip(runs, strides)) + 1
    frames = []
    for f in range(n_frames):
        panels = [draw_panel(st[min(len(st) - 1, f * s)], size, (nm if len(nm) <= 27 else nm[:25] + ".."), a.cell, font, H)
                  for (st, _), s, (nm, size, _) in zip(runs, strides, players)]
        W = sum(p.width for p in panels) + 8 * (len(panels) - 1)
        canvas = Image.new("RGB", (W, H), BG)
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width + 8
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=32))
    frames.extend([frames[-1]] * 16)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames, seed {seed}): {[n for n, _, _ in players]}; steps {[len(st) - 1 for st, _ in runs]}")


if __name__ == "__main__":
    main()

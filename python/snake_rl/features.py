# Mirrors basicFeatures / basicStateIndex / gridObservation in game/index.html.
import numpy as np

from .core import DIRS


def basic_features(g):
    hx, hy = g.body[0]
    f = [0] * 11
    for i, d in enumerate((g.dir, (g.dir + 1) % 4, (g.dir + 3) % 4)):
        f[i] = 1 if g.blocked(hx + DIRS[d][0], hy + DIRS[d][1]) else 0
    f[3 + g.dir] = 1
    if g.food:
        fx, fy = g.food
        f[7], f[8], f[9], f[10] = int(fx < hx), int(fx > hx), int(fy < hy), int(fy > hy)
    return f


def basic_index(g):
    return sum(b << i for i, b in enumerate(basic_features(g)))


def grid_obs(g):
    # ch0 body recency (head 1 → tail 1/L), ch1 head, ch2 food, ch3 wall ring; padded by 1 cell.
    H, W = g.h + 2, g.w + 2
    x = np.zeros((4, H, W), dtype=np.float32)
    L = len(g.body)
    for i, (bx, by) in enumerate(g.body):
        x[0, by + 1, bx + 1] = (L - i) / L
    hx, hy = g.body[0]
    x[1, hy + 1, hx + 1] = 1
    if g.food:
        x[2, g.food[1] + 1, g.food[0] + 1] = 1
    x[3] = 1
    x[3, 1:-1, 1:-1] = 0
    return x


def obs_digest(x):
    # same weighting as tools/make_trace.js::obsDigest
    flat = np.asarray(x, dtype=np.float64).ravel()
    w = (np.arange(flat.size) % 97) + 1
    return float((flat * w).sum())


def win_obs(g, reach=False):
    # Same layout as TorchSnakeVecEnv._win (and gridObservationWin in game/index.html): fixed 14x14 canvas, board at offset 1.
    C, S = 14, g.w
    x = np.zeros((6 if reach else 5, C, C), dtype=np.float32)
    L = len(g.body)
    for i, (bx, by) in enumerate(g.body):
        x[0, by + 1, bx + 1] = (L - i) / L
    hx, hy = g.body[0]
    x[1, hy + 1, hx + 1] = 1
    if g.food:
        x[2, g.food[1] + 1, g.food[0] + 1] = 1
    x[3] = 1
    x[3, 1:S + 1, 1:S + 1] = 0
    x[4, 1:S + 1, 1:S + 1] = L / (S * S)
    if reach:
        free = np.zeros((S, S), dtype=bool)
        for yy in range(S):
            for xx in range(S):
                free[yy, xx] = not g.occ[yy * S + xx]
        seen = np.zeros((S, S), dtype=bool)
        stack = [(hx, hy)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in DIRS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < S and 0 <= ny < S and free[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        x[5, 1:S + 1, 1:S + 1] = seen
    return x


def trap_danger(g):
    """Scalar twin of TorchSnakeVecEnv.danger_trap (and trapDanger in game/index.html): [straight, right, left] -> True if the move
    kills at once or leads into a region too small for the snake (time-aware flood fill, see the vec env for the rule)."""
    S, L = g.w, len(g.body)
    age = {}
    for i, (bx, by) in enumerate(g.body):
        age[(bx, by)] = L - i
    hx, hy = g.body[0]
    inst, areas, cands = [], [], []
    for d in (g.dir, (g.dir + 1) % 4, (g.dir + 3) % 4):
        nx, ny = hx + DIRS[d][0], hy + DIRS[d][1]
        dead = nx < 0 or ny < 0 or nx >= S or ny >= S or age.get((nx, ny), 0) > 1
        inst.append(dead)
        cands.append((nx, ny))
        if dead:
            areas.append(0)
            continue
        reach = {(nx, ny)}
        for m in range(2, S * S // 2 + 3):
            new = set(reach)
            for (cx, cy) in reach:
                for dx, dy in DIRS:
                    q = (cx + dx, cy + dy)
                    if 0 <= q[0] < S and 0 <= q[1] < S and age.get(q, 0) <= m:
                        new.add(q)
            reach = new
        areas.append(len(reach))
    trap = []
    for j in range(3):
        eats = bool(g.food) and cands[j] == tuple(g.food)
        trap.append((not inst[j]) and areas[j] < L + (1 if eats else 0))
    ok = [not (inst[j] or trap[j]) for j in range(3)]
    if not any(ok):
        ok = [not inst[j] for j in range(3)]
    return [not o for o in ok]

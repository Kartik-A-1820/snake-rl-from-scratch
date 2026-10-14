# Line-for-line mirror of the engine in game/index.html (SNAKE CORE block).
# The JS file is the source of truth; tests/test_parity.py replays JS traces here and must match exactly.
from collections import deque

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # R, D, L, U, clockwise: right turn = +1
STRAIGHT, RIGHT, LEFT = 0, 1, 2
M32 = 0xFFFFFFFF


class Rng:
    # mulberry32; Python ints are unbounded, so every op is masked to 32 bits to match JS.
    def __init__(self, seed):
        self.s = seed & M32

    def next_u32(self):
        self.s = (self.s + 0x6D2B79F5) & M32
        t = self.s
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32
        return (t ^ (t >> 14)) & M32

    def int(self, n):
        return (self.next_u32() * n) >> 32


class SnakeGame:
    def __init__(self, width=12, height=12, seed=0, idle_limit=100):
        self.w, self.h, self.idle_limit = width, height, idle_limit
        self.rng = Rng(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = Rng(seed)
        cx, cy = self.w // 2, self.h // 2
        self.body = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])  # head first
        self.occ = bytearray(self.w * self.h)
        for x, y in self.body:
            self.occ[y * self.w + x] = 1
        self.dir = 0
        self.score = self.steps = self.since_food = 0
        self.done = self.won = False
        self.cause = None
        self.place_food()
        return self

    def place_food(self):
        free = [(x, y) for y in range(self.h) for x in range(self.w) if not self.occ[y * self.w + x]]
        self.food = free[self.rng.int(len(free))] if free else None

    def blocked(self, x, y):
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return True
        tx, ty = self.body[-1]
        # the tail vacates this tick, so it is not an obstacle
        return self.occ[y * self.w + x] == 1 and not (x == tx and y == ty)

    def step(self, action):
        if self.done:
            raise RuntimeError("step() after episode end; call reset()")
        if action == RIGHT:
            self.dir = (self.dir + 1) % 4
        elif action == LEFT:
            self.dir = (self.dir + 3) % 4
        dx, dy = DIRS[self.dir]
        nx, ny = self.body[0][0] + dx, self.body[0][1] + dy
        self.steps += 1
        self.since_food += 1

        if self.blocked(nx, ny):
            self.done = True
            self.cause = "wall" if (nx < 0 or ny < 0 or nx >= self.w or ny >= self.h) else "self"
            return -1.0, True, False, self.cause
        eating = self.food is not None and nx == self.food[0] and ny == self.food[1]
        if not eating:
            tx, ty = self.body.pop()
            self.occ[ty * self.w + tx] = 0
        self.body.appendleft((nx, ny))
        self.occ[ny * self.w + nx] = 1

        if eating:
            self.score += 1
            self.since_food = 0
            if len(self.body) == self.w * self.h:
                self.done = self.won = True
                self.food = None
                self.cause = "win"
                return 1.0, True, False, "win"
            self.place_food()
            return 1.0, False, False, "food"
        # truncation, not termination: learners must still bootstrap through it (Pardo et al. 2018)
        if self.since_food > self.idle_limit * len(self.body):
            self.done = True
            self.cause = "timeout"
            return 0.0, False, True, "timeout"
        return 0.0, False, False, None

    def relative_from_absolute(self, abs_dir):
        d = (abs_dir - self.dir + 4) % 4
        return RIGHT if d == 1 else LEFT if d == 3 else STRAIGHT


def greedy_action(g):
    hx, hy = g.body[0]
    best, best_d = 0, float("inf")
    for a in (0, 1, 2):
        d = g.dir if a == 0 else (g.dir + 1) % 4 if a == 1 else (g.dir + 3) % 4
        nx, ny = hx + DIRS[d][0], hy + DIRS[d][1]
        if g.blocked(nx, ny):
            continue
        dist = abs(nx - g.food[0]) + abs(ny - g.food[1]) if g.food else 0
        if dist < best_d:
            best_d, best = dist, a
    return best

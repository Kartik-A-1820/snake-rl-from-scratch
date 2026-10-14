# N Snake games stepped together as tensors, on CPU or CUDA. Same rules and same RNG stream per env as
# the scalar engine (tests/test_parity.py checks it step for step), so results transfer to the browser.
#
# The body is stored as an "age" grid: each body cell holds how many more ticks it stays occupied
# (head = length, tail = 1). Moving = decrement everything and write the new head; eating = skip the
# decrement. That turns the deque of the scalar engine into a handful of batched tensor ops.
import torch
import torch.nn.functional as F

M32 = 0xFFFFFFFF
CAUSES = ("", "wall", "self", "timeout", "win")


def _imul(a, b):
    # (a*b) mod 2^32 for uint32 values held in int64, split so the product never exceeds 2^49
    return ((a & 0xFFFF) * b + ((((a >> 16) * b) & 0xFFFF) << 16)) & M32


def mulberry32(s):
    s = (s + 0x6D2B79F5) & M32
    t = _imul(s ^ (s >> 15), s | 1)
    t = t ^ ((t + _imul(t ^ (t >> 7), t | 61)) & M32)
    return s, (t ^ (t >> 14)) & M32


class TorchSnakeVecEnv:
    def __init__(self, n, size=12, seeds=None, device="cpu", idle_limit=100, obs="grid",
                 shaping="none", shape_k=1.0, gamma=0.99, timeout_penalty=0.0, win_reward=0.0):
        # shaping: "none" | "naive" (+-k per move toward/away from food) | "potential" (gamma*phi(s') - phi(s),
        # phi = -k * manhattan(head, food) / (2*size); Ng, Harada & Russell 1999). timeout_penalty is added on a
        # starvation truncation (this one is NOT policy-invariant: it deliberately changes what is optimal).
        self.n, self.size, self.idle_limit, self.obs_type = n, size, idle_limit, obs
        self.shaping, self.shape_k, self.gamma, self.timeout_penalty = shaping, shape_k, gamma, timeout_penalty
        self.win_reward = win_reward   # extra reward for filling the board (tryout 05), on top of the +1 for the last food
        self.device = dev = torch.device(device)
        S = size
        self.dirs = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=dev)
        self.turn = torch.tensor([0, 1, 3], device=dev)
        self.ar = torch.arange(n, device=dev)
        c = S // 2
        self.start_age = torch.zeros(S, S, dtype=torch.int32, device=dev)
        self.start_age[c, c], self.start_age[c, c - 1], self.start_age[c, c - 2] = 3, 2, 1
        self.start_head = torch.tensor([c, c], device=dev)
        self._cross = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], device=dev).view(1, 1, 3, 3)
        self.wall = torch.ones(S + 2, S + 2, device=dev)
        self.wall[1:-1, 1:-1] = 0
        z = lambda: torch.zeros(n, dtype=torch.long, device=dev)
        self.age = torch.zeros(n, S, S, dtype=torch.int32, device=dev)
        self.head, self.food = torch.zeros(n, 2, dtype=torch.long, device=dev), torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.dir, self.length, self.score, self.steps, self.since_food = z(), z(), z(), z(), z()
        self.reset(seeds)

    def reset(self, seeds=None):
        seeds = torch.arange(self.n) if seeds is None else torch.as_tensor(seeds)
        self.rng = (seeds.to(torch.long) & M32).to(self.device)
        self._reset(torch.ones(self.n, dtype=torch.bool, device=self.device))
        return self.obs()

    def _reset(self, m):
        # mask-only ops (no .any()/.nonzero()) so CUDA never has to sync with the host here
        self.age = torch.where(m.view(-1, 1, 1), self.start_age, self.age)
        self.head = torch.where(m.unsqueeze(1), self.start_head, self.head)
        for name, v in (("dir", 0), ("length", 3), ("score", 0), ("steps", 0), ("since_food", 0)):
            setattr(self, name, torch.where(m, v, getattr(self, name)))
        self._place_food(m)

    def _place_food(self, m):
        # k-th free cell in row-major order, k = floor(u * nfree / 2^32): identical to the scalar engine
        free = (self.age == 0).view(self.n, -1)
        s_new, u = mulberry32(self.rng)
        self.rng = torch.where(m, s_new, self.rng)  # only envs that actually place food consume a draw
        k = (u * free.sum(1)) >> 32
        pos = (free.long().cumsum(1) > k.unsqueeze(1)).float().argmax(1)
        new = torch.stack([pos % self.size, pos // self.size], 1)
        self.food = torch.where(m.unsqueeze(1), new, self.food)

    @torch.no_grad()
    def step(self, actions, lazy_next=False):
        # lazy_next=True (PPO, evaluation): do not build the pre-reset next observation and next-state danger mask unless a timeout
        # truncation needs the next observation (info["trunc_any"] says so). Halves the observation cost per step; learning is unchanged.
        n, S = self.n, self.size
        self.dir = (self.dir + self.turn[actions.to(self.device).long()]) % 4
        nxt = self.head + self.dirs[self.dir]
        nx, ny = nxt[:, 0], nxt[:, 1]
        oob = (nx < 0) | (ny < 0) | (nx >= S) | (ny >= S)
        cx, cy = nx.clamp(0, S - 1), ny.clamp(0, S - 1)
        cell = self.age[self.ar, cy, cx]
        self.steps += 1
        self.since_food += 1
        d0 = (self.head - self.food).abs().sum(1).float() if self.shaping != "none" else None  # distance before the move

        hit_self = ~oob & (cell > 1)  # age 1 = tail, which vacates this tick
        dead = oob | hit_self
        alive = ~dead
        eating = alive & (nx == self.food[:, 0]) & (ny == self.food[:, 1])
        move = alive & ~eating

        self.age = self.age - (move.view(n, 1, 1) & (self.age > 0)).int()
        self.length = self.length + eating.long()
        flat = self.age.view(n, -1)
        fi = (cy * S + cx).unsqueeze(1)
        flat.scatter_(1, fi, torch.where(alive.unsqueeze(1), self.length.int().unsqueeze(1), flat.gather(1, fi)))
        self.head = torch.where(alive.unsqueeze(1), nxt, self.head)
        self.score = self.score + eating.long()
        self.since_food = torch.where(eating, 0, self.since_food)

        won = eating & (self.length == S * S)
        trunc = move & (self.since_food > self.idle_limit * self.length)
        if self.shaping == "naive":
            # toward/away judged against the food that was on the board when the move was chosen
            d1 = (self.head - self.food).abs().sum(1).float()
            shaped = torch.where(dead, 0.0, self.shape_k * torch.sign(d0 - d1))
        self._place_food(eating & ~won)

        reward = torch.where(dead, -1.0, torch.where(eating, 1.0, 0.0))
        if self.shaping == "naive":
            reward = reward + shaped
        elif self.shaping == "potential":
            # phi(s') uses the NEW food after an eat (s' includes it); phi = 0 on terminal states (death, win).
            # Timeouts are truncations, not terminals: s' is a real state, so phi(s') is kept and the bootstrap continues.
            phi0 = -self.shape_k * d0 / (2 * S)
            phi1 = -self.shape_k * (self.head - self.food).abs().sum(1).float() / (2 * S)
            phi1 = torch.where(dead | won, 0.0, phi1)
            reward = reward + self.gamma * phi1 - phi0
        if self.timeout_penalty:
            reward = reward - self.timeout_penalty * trunc.float()
        if self.win_reward:
            reward = reward + self.win_reward * won.float()
        terminated = dead | won
        cause = torch.where(won, 4, torch.where(trunc, 3, torch.where(hit_self, 2, torch.where(oob, 1, 0))))
        info = {"score": self.score.clone(), "length": self.length.clone(), "cause": cause, "won": won}
        if lazy_next:
            need = trunc & ~terminated
            info["trunc_any"] = bool(need.any())          # one host sync, needed by the caller anyway
            info["next_obs"] = self.obs() if info["trunc_any"] else None
        else:
            info["next_obs"] = self.obs()
            info["next_danger"] = self.danger()           # pre-reset, like next_obs
        self._reset(terminated | trunc)
        return self.obs(), reward, terminated, trunc, info

    def obs(self):
        if self.obs_type in ("win5", "win6"):
            return self._win(reach=self.obs_type == "win6")
        return self._grid() if self.obs_type == "grid" else self._basic()

    def _win(self, reach=False):
        # Fixed 14x14 canvas, so one network plays every board up to 12x12 (the board sits at offset 1, everything outside
        # is wall). Channels: 0 tail order (age/length: how long until each body cell frees), 1 head, 2 food, 3 wall,
        # 4 length / cells (constant plane), 5 (win6 only) cells reachable from the head through empty cells.
        n, S, C = self.n, self.size, 14
        x = torch.zeros(n, 6 if reach else 5, C, C, device=self.device)
        x[:, 0, 1:S + 1, 1:S + 1] = self.age.float() / self.length.view(n, 1, 1).float()
        x[:, 1, 1:S + 1, 1:S + 1] = F.one_hot(self.head[:, 1] * S + self.head[:, 0], S * S).view(n, S, S).float()
        x[:, 2, 1:S + 1, 1:S + 1] = F.one_hot(self.food[:, 1] * S + self.food[:, 0], S * S).view(n, S, S).float()
        x[:, 3] = 1.0
        x[:, 3, 1:S + 1, 1:S + 1] = 0.0
        x[:, 4, 1:S + 1, 1:S + 1] = (self.length.float() / (S * S)).view(n, 1, 1)
        if reach:
            free = (self.age == 0).float()
            r = x[:, 1, 1:S + 1, 1:S + 1].clone()
            k = self._cross
            for _ in range(max(2, S * S // 2)):   # flood fill: one cell per step through empty cells
                r = torch.clamp(F.conv2d(r.unsqueeze(1), k, padding=1).squeeze(1), 0, 1) * free + x[:, 1, 1:S + 1, 1:S + 1]
            x[:, 5, 1:S + 1, 1:S + 1] = torch.clamp(r, 0, 1) * free
        return x

    def _grid(self):
        n, S = self.n, self.size
        x = torch.zeros(n, 4, S + 2, S + 2, device=self.device)
        x[:, 0, 1:-1, 1:-1] = self.age.float() / self.length.view(n, 1, 1).float()
        x[:, 1, 1:-1, 1:-1] = F.one_hot(self.head[:, 1] * S + self.head[:, 0], S * S).view(n, S, S).float()
        x[:, 2, 1:-1, 1:-1] = F.one_hot(self.food[:, 1] * S + self.food[:, 0], S * S).view(n, S, S).float()
        x[:, 3] = self.wall
        return x

    def danger(self):
        # (n, 3) bool: does relative action [straight, right, left] kill the snake on the very next tick?
        # Same rule as the step (wall, or a body cell other than the tail, which vacates).
        S = self.size
        rel = torch.stack([self.dir, (self.dir + 1) % 4, (self.dir + 3) % 4], 1)
        cand = self.head.unsqueeze(1) + self.dirs[rel]
        x, y = cand[..., 0], cand[..., 1]
        oob = (x < 0) | (y < 0) | (x >= S) | (y >= S)
        cell = self.age[self.ar.unsqueeze(1), y.clamp(0, S - 1), x.clamp(0, S - 1)]
        return oob | (cell > 1)

    def danger_trap(self):
        """(n, 3) bool danger mask that also blocks moves into a trap (tryout 05 "shielded" agent). A move is a trap when, after it,
        the cells the head can still reach cannot hold the snake: time-aware flood fill (a body cell is enterable at move m once its
        age <= m, because the tail keeps freeing cells), K = cells/2 + 2 expansions, area < length (+1 if the move eats).
        Fallback: if every move is a trap, only instant deaths are blocked; if every move kills, nothing is blocked
        (same convention as the instant-death mask)."""
        n, S = self.n, self.size
        inst = self.danger()
        rel = torch.stack([self.dir, (self.dir + 1) % 4, (self.dir + 3) % 4], 1)
        cand = (self.head.unsqueeze(1) + self.dirs[rel]).clamp(0, S - 1)          # (n, 3, 2) as x, y
        r = F.one_hot(cand[..., 1] * S + cand[..., 0], S * S).view(n, 3, S, S).float() * (~inst).float().view(n, 3, 1, 1)
        age = self.age.unsqueeze(1).expand(n, 3, S, S)
        r = r.reshape(n * 3, 1, S, S)
        for m in range(2, S * S // 2 + 3):
            allowed = (age <= m).float().reshape(n * 3, 1, S, S)
            r = torch.clamp(F.conv2d(r, self._cross, padding=1), 0, 1) * allowed
        area = r.view(n, 3, S * S).sum(2)
        eats = (cand[..., 0] == self.food[:, 0:1]) & (cand[..., 1] == self.food[:, 1:2])
        need = (self.length.view(n, 1) + eats.long()).float()
        trap = (~inst) & (area < need)
        ok = ~(inst | trap)
        ok2 = ~inst
        ok = torch.where(ok.any(1, keepdim=True), ok, ok2)
        return ~ok

    def _basic(self):
        danger = self.danger().float()
        hx, hy, fx, fy = self.head[:, 0], self.head[:, 1], self.food[:, 0], self.food[:, 1]
        food = torch.stack([fx < hx, fx > hx, fy < hy, fy > hy], 1).float()
        return torch.cat([danger, F.one_hot(self.dir, 4).float(), food], 1)

    def body(self, i):
        # body of env i as [(x, y), ...] head first; for tests and debugging only
        ys, xs = torch.nonzero(self.age[i] > 0, as_tuple=True)
        order = torch.argsort(self.age[i][ys, xs], descending=True)
        return [(int(xs[j]), int(ys[j])) for j in order]


@torch.no_grad()
def evaluate(policy, seeds, size=12, device="cpu", max_steps=50_000, obs="grid", masked=False, trap=False):
    # Greedy evaluation: one episode per seed, all run in parallel; policy(obs) -> actions tensor.
    # masked=True: policy(obs, danger) with danger (n, 3) bool = moves that kill on the next tick (action masking).
    env = TorchSnakeVecEnv(len(seeds), size, seeds=seeds, device=device, obs=obs)
    x = env.obs()
    finished = torch.zeros(len(seeds), dtype=torch.bool, device=env.device)
    scores = torch.zeros(len(seeds), dtype=torch.long, device=env.device)
    causes = torch.zeros(len(seeds), dtype=torch.long, device=env.device)
    for t in range(max_steps):
        x, _, term, trunc, info = env.step(policy(x, env.danger_trap() if trap else env.danger()) if masked else policy(x), lazy_next=True)
        newly = (term | trunc) & ~finished
        scores = torch.where(newly, info["score"], scores)
        causes = torch.where(newly, info["cause"], causes)
        finished |= newly
        if t % 50 == 0 and bool(finished.all()):  # checking every step would sync the GPU every step
            break
    # safety net: anything still running after max_steps counts as a timeout at its current score
    scores = torch.where(finished, scores, env.score)
    causes = torch.where(finished, causes, 3)
    s = scores.cpu().float()
    c = causes.cpu()
    q = torch.quantile(s, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    mid = s.sort().values[int(len(s) * 0.25): int(round(len(s) * 0.75))]
    return {
        "n": len(seeds), "mean": float(s.mean()), "median": float(q[2]), "p10": float(q[0]), "p90": float(q[4]),
        "max": int(s.max()), "iqm": float(mid.mean()), "wins": int((c == 4).sum()),
        "causes": {CAUSES[k]: int((c == k).sum()) for k in range(1, 5) if (c == k).any()},
        "scores": scores.cpu().tolist(), "cause_ids": c.tolist(),
    }

# Gymnasium wrapper around the scalar engine. Use it for debugging, tests and single-env algorithms;
# use vec_env.TorchSnakeVecEnv for fast training.
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import SnakeGame
from .features import basic_features, grid_obs


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=12, obs="grid", idle_limit=100, shaping=None, gamma=0.99, shape_k=1.0, timeout_penalty=0.0):
        # shaping: None | "naive" (+-k per move toward/away from food) | "potential" (same as TorchSnakeVecEnv)
        self.size, self.obs_type, self.shaping, self.gamma = size, obs, shaping, gamma
        self.shape_k, self.timeout_penalty = shape_k, timeout_penalty
        self.game = SnakeGame(size, size, seed=0, idle_limit=idle_limit)
        self.action_space = spaces.Discrete(3)  # straight, right, left (relative to heading)
        if obs == "grid":
            self.observation_space = spaces.Box(0.0, 1.0, (4, size + 2, size + 2), np.float32)
        else:
            self.observation_space = spaces.Box(0.0, 1.0, (11,), np.float32)

    def _obs(self):
        if self.obs_type == "grid":
            return grid_obs(self.game)
        return np.asarray(basic_features(self.game), dtype=np.float32)

    def _phi(self):
        # potential = negative normalised distance to food (Ng, Harada & Russell 1999): shaping of the form
        # gamma*phi(s') - phi(s) cannot change the optimal policy.
        g = self.game
        if g.food is None:
            return 0.0
        hx, hy = g.body[0]
        return -self.shape_k * (abs(hx - g.food[0]) + abs(hy - g.food[1])) / (2 * self.size)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # seed → restart the engine RNG; no seed → continue its stream (matches the JS game and the vec env)
        self.game.reset(seed)
        return self._obs(), {}

    def step(self, action):
        phi0 = self._phi() if self.shaping == "potential" else 0.0
        g = self.game
        d0 = abs(g.body[0][0] - g.food[0]) + abs(g.body[0][1] - g.food[1]) if g.food else 0
        food0 = g.food
        reward, terminated, truncated, event = g.step(int(action))
        if self.shaping == "potential":
            phi1 = 0.0 if terminated else self._phi()
            reward += self.gamma * phi1 - phi0
        elif self.shaping == "naive" and food0 and not (terminated and g.cause != "win"):
            d1 = abs(g.body[0][0] - food0[0]) + abs(g.body[0][1] - food0[1])
            reward += self.shape_k * ((d0 > d1) - (d0 < d1))
        if truncated:
            reward -= self.timeout_penalty
        info = {"event": event, "score": self.game.score}
        if terminated or truncated:
            info["cause"] = self.game.cause
            info["won"] = self.game.won
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        g = self.game
        rows = [["." for _ in range(g.w)] for _ in range(g.h)]
        for x, y in g.body:
            rows[y][x] = "o"
        rows[g.body[0][1]][g.body[0][0]] = "@"
        if g.food:
            rows[g.food[1]][g.food[0]] = "*"
        return "\n".join("".join(r) for r in rows)


gym.register(id="Snake-v0", entry_point="snake_rl.env:SnakeEnv")

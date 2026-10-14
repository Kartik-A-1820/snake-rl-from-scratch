# Registers the non-RL Hamiltonian-cycle agent (models/05_hamiltonian/): the upper bound for the win challenge.
#   python python/publish_hamiltonian.py
from snake_rl import registry
from snake_rl.core import SnakeGame
from snake_rl.hamilton import ham_action


def play(size, seed):
    g, acts = SnakeGame(size, size, seed=seed), []
    while not g.done:
        a = ham_action(g)
        acts.append(a)
        g.step(a)
    return g, "".join(map(str, acts))


def main():
    by_size, replays = [], []
    for size in (6, 8, 10, 12):
        scores, causes = [], {}
        for seed in registry.TEST_SEEDS:
            g, acts = play(size, seed)
            scores.append(g.score)
            causes[g.cause] = causes.get(g.cause, 0) + 1
            if seed == registry.TEST_SEEDS[0]:
                replays.append({"seed": seed, "width": size, "height": size, "actions": acts, "score": g.score, "cause": g.cause})
        st = registry.summarize_scores(scores, causes)
        by_size.append({"size": size, "win_rate": st["wins"] / st["n"], "mean": st["mean"], "max": st["max"]})
        print(size, st["wins"], "/", st["n"], "wins, mean score", st["mean"])
        if size == 12:
            test = st
    meta = {"name": "Hamiltonian cycle (planning, not RL)", "tryout": 5, "family": "planning", "non_rl": True,
            "description": "Follows a fixed Hamiltonian cycle of the board. Needs the rules, not learning: it fills the board on every seed. "
                           "The upper bound the learned agents are measured against.",
            "obs": "none", "size": 12, "size_locked": False, "params": 0, "onnx_cpu_latency_us": None, "select_score": None,
            "requires": "the game rules (even board, at least 6x6); no learning", "test": {k: test[k] for k in ("n", "mean", "std", "median", "p10", "p90", "max", "wins", "causes")},
            "by_size": by_size, "train": None, "curve": None, "replays": replays, "notes": "../notes/05_win_challenge.html",
            "source_ckpt": "snake_rl/hamilton.py"}
    registry.publish("05_hamiltonian", {"type": "hamiltonian", "size": 12}, meta)
    print("published 05_hamiltonian")


if __name__ == "__main__":
    main()

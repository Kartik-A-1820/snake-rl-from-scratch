# Final registry pass, run once all training is done:
#   * publishes the best seed of each tryout-03 component arm as its own agent (for the game's Compare mode),
#   * publishes the final PPO agent (tryout 04): the arm with the higher mean over seeds, best-validation checkpoint,
#   * writes "seed_stats" (mean and sample sd over training seeds, with the scope) into the meta of agents that have seed groups,
#   * rebuilds models/registry.js.
#   python python/finalize_registry.py
import json
import statistics
from pathlib import Path

from snake_rl import registry
from snake_rl.export import export_checkpoint

ROOT = Path(__file__).resolve().parents[1]
RES, RUNS = ROOT / "results", ROOT / "models" / "_runs"


def summary(phase, run):
    p = RES / phase / run / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def best_means(phase, runs):
    out = []
    for r in runs:
        s = summary(phase, r)
        if s and s.get("best"):
            out.append((r, s["best"]["test"]["mean"], s["best"]["val_mean"]))
    return out


def stats(means, scope):
    v = [m for _, m, _ in means]
    return {"n": len(v), "mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0, "min": min(v), "max": max(v),
            "per_seed": [round(x, 2) for x in v], "scope": scope}


def set_seed_stats(algo_id, st):
    p = registry.MODELS / algo_id / "meta.json"
    if not p.exists():
        print(f"  (no {algo_id} to attach seed stats to)")
        return
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["seed_stats"] = st
    p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"  {algo_id}: seeds {st['per_seed']} -> {st['mean']:.2f} ± {st['std']:.2f}")


def main():
    # --- tryout 03: component arms (1 M steps, best seed of each) ---
    arms = {"duel": ("03_dqn_dueling", "DQN + dueling (1 M steps)", "Tryout 02's DQN with a dueling head. 1 M steps, best of 3 seeds."),
            "duel_per": ("03_dqn_dueling_per", "DQN + dueling + PER (1 M steps)", "Dueling head and prioritized replay. 1 M steps, best of 3 seeds."),
            "mask": ("03_dqn_mask", "DQN + action mask (1 M steps)", "Tryout 02's DQN with action masking only. 1 M steps, best of 3 seeds.")}
    for arm, (algo, name, desc) in arms.items():
        runs = [f"t03_{arm}_s{s}" for s in range(3)]
        means = best_means("phase2_dqn", runs)
        if not means:
            continue
        best_run = max(means, key=lambda x: x[2])[0]   # choose by validation, never by test
        export_checkpoint(RUNS / f"{best_run}_best.pt", algo, name, 3, "../notes/03_dqn_upgrades.html", desc, run=best_run)
        set_seed_stats(algo, stats(means, "1 M-step runs"))
    full = best_means("phase2_dqn", [f"t03_full_s{s}" for s in range(3)])
    if full:
        set_seed_stats("03_dqn_plus", stats(full, "1 M-step runs of the same combination"))
    base = best_means("phase2_dqn", ["diag_B_grid_more_updates_n3", "seed1_arm1_baseline", "seed2_arm1_baseline"])
    if base:
        set_seed_stats("02_dqn_cnn", stats(base, "1 M-step runs of the same setup"))

    # --- tryout 04: the better PPO arm ---
    groups = {arm: best_means("phase4_ppo", [f"t04_ppo_{arm}_s{s}" for s in range(3)]) for arm in ("base", "mask")}
    groups = {a: g for a, g in groups.items() if g}
    if groups:
        arm = max(groups, key=lambda a: statistics.mean(m for _, m, _ in groups[a]))
        best_run = max(groups[arm], key=lambda x: x[2])[0]
        name = "PPO (CNN, action mask)" if arm == "mask" else "PPO (CNN)"
        export_checkpoint(RUNS / f"{best_run}_best.pt", "04_ppo", name, 4, "../notes/04_ppo.html",
                          "PPO actor-critic" + (" with invalid-action masking" if arm == "mask" else "") + " on the board image, 10 M steps. "
                          f"Best-validation checkpoint of the best arm (seed run {best_run}).", run=best_run)
        set_seed_stats("04_ppo", stats(groups[arm], "10 M-step runs of the same arm"))
    registry.rebuild_registry()
    print("registry rebuilt")


if __name__ == "__main__":
    main()

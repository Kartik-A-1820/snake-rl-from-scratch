# Collects run summaries into notes/tryouts/<NN>/results.json for the notes builder (notes/build_notes.py).
#   python python/make_notes_data.py            # both tryouts 03 and 04
#   python python/make_notes_data.py --only 03
# 03: the ablation arms (per training seed) + the long run of the full combination (same shape as tryout 02's results.json).
# 04: the PPO arms + the best PPO run + the measured parallel-throughput benchmarks.
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
RES = ROOT / "results"
NOTES = ROOT / "notes" / "tryouts"


def load(phase, name):
    p = RES / phase / name / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    from snake_rl.export import clean_epochs   # epochs inflated by the PC sleeping must not count as training time
    s["epochs"] = clean_epochs(s["epochs"])
    return s


def in_progress_summary(phase, name):
    """A summary-shaped dict for a run that is still training (no summary.json yet), rebuilt from its config, per-epoch metrics and
    best checkpoint, so the notes and the README can show the current numbers."""
    d = RES / phase / name
    if not (d / "metrics.jsonl").exists() or not (d / "config.json").exists():
        return None
    import torch
    from snake_rl.export import clean_epochs
    metrics = [json.loads(line) for line in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cps = [m["checkpoint"] for m in metrics if "checkpoint" in m]
    best_pt = ROOT / "models" / "_runs" / f"{name}_best.pt"
    if not cps or not best_pt.exists():
        return None
    ck = torch.load(best_pt, map_location="cpu", weights_only=False)
    return {"name": name, "config": json.loads((d / "config.json").read_text(encoding="utf-8")), "epochs": clean_epochs(metrics), "checkpoints": cps,
            "best": {"epoch": ck["epoch"], "val_mean": ck["val_mean"], "test": ck["test"]}, "first_win": None, "in_progress": True}


def run_entry(s, seed):
    if not s or not s.get("best"):
        return None
    return {"seed": seed, "run": s["name"], "best": s["best"],
            "curve": [[c["epoch"], c["test"]["mean"]] for c in s["checkpoints"]],
            "seconds": round(sum(e["seconds"] for e in s["epochs"])), "steps_per_s": round(sum(e["steps_per_s"] for e in s["epochs"]) / len(s["epochs"]))}


def arm(arm_id, label, components, runs):
    runs = [r for r in runs if r]
    return {"id": arm_id, "label": label, "components": components, "runs": runs} if runs else None


def tryout03():
    base_runs = ["diag_B_grid_more_updates_n3", "seed1_arm1_baseline", "seed2_arm1_baseline"]
    arms = [arm("baseline", "baseline (Double, 3-step)", ["double"], [run_entry(load("phase2_dqn", n), i) for i, n in enumerate(base_runs)])]
    for aid, label, comps in (("duel", "+ dueling", ["double", "dueling"]), ("duel_per", "+ dueling + PER", ["double", "dueling", "per"]),
                              ("mask", "+ masking only", ["double", "mask"]), ("full", "+ dueling + PER + masking", ["double", "dueling", "per", "mask"])):
        arms.append(arm(aid, label, comps, [run_entry(load("phase2_dqn", f"t03_{aid}_s{s}"), s) for s in range(3)]))
    long = load("phase2_dqn", "t03_long_full") or {}
    out = {k: long.get(k) for k in ("name", "config", "epochs", "checkpoints", "best", "replays")}
    out.update(tryout="03_dqn_upgrades", arms=[a for a in arms if a], benchFile="results/bench/bench.json",
               baselines12={"random": 0.168, "greedy_heuristic": 20.416, "tabular_q": 20.08})
    return out


def tryout04():
    arms = []
    best = None
    for aid, label in (("base", "PPO"), ("mask", "PPO + action masking")):
        runs = [run_entry(load("phase4_ppo", f"t04_ppo_{aid}_s{s}"), s) for s in range(3)]
        a = arm(aid, label, [], runs)
        if a:
            arms.append(a)
            for r in a["runs"]:
                if best is None or r["best"]["val_mean"] > best[0]:
                    best = (r["best"]["val_mean"], r["run"])
    out = {"tryout": "04_ppo", "arms": arms}
    if best:
        out.update({k: load("phase4_ppo", best[1]).get(k) for k in ("name", "config", "epochs", "checkpoints", "best")})
    par = {}
    for fam in ("ppo", "dqn"):
        p = RES / "bench" / f"parallel_{fam}.json"
        if p.exists():
            par[fam] = json.loads(p.read_text(encoding="utf-8"))
    out["parallel"] = par
    out["baselines12"] = {"random": 0.168, "greedy_heuristic": 20.416, "tabular_q": 20.08}
    return out


RUN_INFO = {   # run name -> (what it is, what it started from)
    "t05_6x6_win5_s0": ("6x6, canvas without the reachable-area channel (control; stopped at 4 M steps)", "scratch"),
    "t05_6x6_win6_s0": ("6x6, with the reachable-area channel (stopped at 6 M steps)", "scratch"),
    "t05_8x8_win6_scratch": ("8x8 from scratch (12 M steps)", "scratch"),
    "t05_8x8_win6_s0": ("8x8 initialised from the 6x6 agent (12 M steps)", "6x6 agent"),
    "t05_8x8_win6_trap_s0": ("8x8 from the 6x6 agent + trap shield (8 M steps)", "6x6 agent"),
    "t05_10x10_win6_scratch": ("10x10 from scratch (16 M steps)", "scratch"),
    "t05_10x10_win6_cont": ("10x10, the run above continued for 16 M more steps", "10x10 agent (16 M steps)"),
    "t05_10x10_win6_trap_s0": ("10x10 from the shielded 8x8 agent (paused at 3 M steps)", "8x8 shielded agent"),
    "t05_12x12_win6_scratch": ("12x12 from scratch (24 M steps)", "scratch"),
    "t05_12x12_ext": ("12x12, the run above continued (extended run, in progress)", "12x12 agent (24 M steps)"),
}


def tryout05():
    """Win challenge: one stage per board size, each with its runs (curriculum fine-tune and the from-scratch control)."""
    stages, replays = {}, []
    for d in sorted((RES / "phase4_ppo").glob("t05_*")):
        s = load("phase4_ppo", d.name) or in_progress_summary("phase4_ppo", d.name)
        if not s or not s.get("best"):
            continue
        c = s["config"]
        init = Path(c["init_from"]).stem.removesuffix("_best") if c.get("init_from") else None
        init_size = (init.split("_")[1] if init and init.startswith("t05_") else ("12x12 (24 M steps)" if init and "12x12" in init else None)) if init else None   # t05_6x6_win5_s0 -> 6x6
        info = RUN_INFO.get(d.name, (d.name, init_size or "scratch"))
        run = {"name": d.name, "label": info[0], "start": info[1], "in_progress": bool(s.get("in_progress")) or d.name == "t05_12x12_ext", "init_from": init_size, "scratch": c["size"] > 6 and not init and "ext" not in d.name, "best": s["best"], "steps": s["epochs"][-1]["env_steps"],
               "curve": [[cp["epoch"], cp["test"]["mean"], cp["test"]["wins"], cp["test"].get("n", 500)] for cp in s["checkpoints"]],
               "first_win": {k: v for k, v in (s.get("first_win") or {}).items() if k != "replays"} or None}
        stages.setdefault(c["size"], []).append(run)
        for rp in ((s.get("first_win") or {}).get("replays") or [])[:1]:
            replays.append(dict(rp, label=f"{d.name} (first win, epoch {s['first_win']['epoch']})"))
    ham = json.loads((ROOT / "models" / "05_hamiltonian" / "meta.json").read_text(encoding="utf-8")) if (ROOT / "models" / "05_hamiltonian" / "meta.json").exists() else {}
    for rp in ham.get("replays", []):
        replays.append(dict(rp, label="Hamiltonian cycle (not RL)"))
    return {"tryout": "05_win_challenge", "stages": [{"size": k, "label": f"{k}x{k}", "runs": v} for k, v in sorted(stages.items())],
            "hamilton": ham.get("by_size"), "replays": replays}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["03", "04", "05"])
    a = ap.parse_args()
    for key, fn, d in (("03", tryout03, "03_dqn_upgrades"), ("04", tryout04, "04_ppo"), ("05", tryout05, "05_win_challenge")):
        if a.only in (None, key):
            data = fn()
            (NOTES / d).mkdir(parents=True, exist_ok=True)
            (NOTES / d / "results.json").write_text(json.dumps(data), encoding="utf-8")
            if "arms" in data:
                print(f"{d}: {len(data['arms'])} arms, runs per arm {[len(x['runs']) for x in data['arms']]}")
            else:
                print(f"{d}: {len(data['stages'])} stages, {len(data['replays'])} replays")


if __name__ == "__main__":
    main()

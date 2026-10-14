# Writes results/plan_tonight.json: the priority-ordered job list for jobq.py (tryout 05 win-challenge curriculum first,
# then the remaining seeds of the tryout 03 / 04 ablations).
#   python python/plan_tonight.py && python python/jobq.py --plan results/plan_tonight.json
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON05 = ["--device", "cuda", "--mask", "--gamma", "0.995", "--win-reward", "10", "--total-steps", "12000000",
            "--epoch-steps", "500000", "--checkpoint-every", "2", "--obs", "win6"]
DQN = ["--device", "cuda", "--total-steps", "1000000", "--checkpoint-every", "5"]
DQN_ARMS = {"duel": ["--dueling"], "duel_per": ["--dueling", "--per"], "mask": ["--mask"], "full": ["--dueling", "--per", "--mask"]}
PPO = ["--device", "cuda", "--total-steps", "10000000", "--epoch-steps", "500000", "--checkpoint-every", "4"]


def stage(size, prev, name=None, scratch=False):
    name = name or f"t05_{size}x{size}_win6_s0"
    args = COMMON05 + ["--size", str(size), "--publish", f"05_ppo_win_{size}x{size}", "--publish-name", f"PPO win challenge, {size}x{size}",
                       "--tryout", "5", "--notes", "../notes/05_win_challenge.html",
                       "--description", f"PPO with an action mask on the size-agnostic canvas, trained to fill a {size}x{size} board"
                                        + ("." if scratch or not prev else f", fine-tuned from the {prev}x{prev} agent.")]
    if scratch:   # the from-scratch control is for the comparison only; it must not replace the registered curriculum agent
        i = args.index("--publish")
        args = args[:i] + args[i + 8:]
    j = {"name": name, "script": "train_ppo.py", "args": args, "peak_mb": 800}
    if prev and not scratch:
        j["args"] += ["--init-from", f"models/_runs/t05_{prev}x{prev}_win6_s0_best.pt"]
        j["requires"] = [f"results/phase4_ppo/t05_{prev}x{prev}_win6_s0/summary.json"]
    return j


def dqn(arm, s):
    return {"name": f"t03_{arm}_s{s}", "script": "train_dqn.py", "args": DQN + DQN_ARMS[arm] + ["--seed", str(s)], "peak_mb": 900}


def ppo(arm, s):
    return {"name": f"t04_ppo_{arm}_s{s}", "script": "train_ppo.py", "args": PPO + (["--mask"] if arm == "mask" else []) + ["--seed", str(s)], "peak_mb": 600}


# The 10x10 / 12x12 win-challenge stages are launched by hand from whichever 8x8 variant actually wins (see tryout 05 notes),
# so they are not queued here. The from-scratch 8x8 control comes first, then the remaining seeds of tryouts 03 and 04.
plan = [stage(8, None, "t05_8x8_win6_scratch", scratch=True)] + [dqn(a, 1) for a in DQN_ARMS] + [ppo("base", 1), ppo("mask", 1)] + [dqn(a, 2) for a in DQN_ARMS] + [ppo("base", 2), ppo("mask", 2)]
(ROOT / "results" / "plan_tonight.json").write_text(json.dumps(plan, indent=1))
print(len(plan), "jobs:", [j["name"] for j in plan])

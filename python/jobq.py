# GPU job queue: launches the next training job only when the machine has headroom, and logs the throughput it sees.
#
# A job is launched when ALL of these hold (checked every --poll seconds):
#   1. fewer than --max-jobs training processes are running (any train_dqn.py / train_ppo.py, whoever started them),
#   2. GPU memory free >= the job's estimated peak + --margin-mb (default 300),
#   3. CPU utilisation (all logical cores) < --cpu-limit,
#   4. GPU utilisation (mean of 3 samples) < --util-limit: if the GPU is already saturated another job only slows the rest.
# --max-jobs is the knee measured by bench_parallel.py: past it more concurrent jobs only slow each other down, because
# the GPU's compute and the CPU cores that feed it saturate long before the 4 GB of VRAM does.
# Jobs whose results/*/<name>/summary.json exists, or whose name is already running, are skipped, so the queue is resumable
# and can be restarted at any time without touching running jobs. A plan entry may list "requires": [paths relative to the repo
# root] that must exist first (e.g. the previous curriculum stage's summary.json); until then later entries launch ahead of it. Every poll appends a row to results/bench/jobq_timeline.csv
# (time, running jobs, GPU util %, GPU mem MB, CPU %, aggregate steps/s) for the throughput notes.
#
#   python python/jobq.py --max-jobs 3                 # the default plan below
#   python python/jobq.py --plan plan.json             # [{"name":..., "script":"train_ppo.py", "args":[...], "peak_mb":700}, ...]
import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv-cuda" / "Scripts" / "python.exe"
RES = ROOT / "results"
DQN_COMMON = ["--device", "cuda", "--total-steps", "1000000", "--checkpoint-every", "5"]
DQN_ARMS = {"duel": ["--dueling"], "duel_per": ["--dueling", "--per"], "mask": ["--mask"], "full": ["--dueling", "--per", "--mask"]}
PPO_COMMON = ["--device", "cuda", "--total-steps", "10000000", "--epoch-steps", "500000", "--checkpoint-every", "4"]
PPO_ARMS = {"base": [], "mask": ["--mask"]}


def default_plan(seeds=(0, 1, 2)):
    """Tryout 03 (DQN ablation) and tryout 04 (PPO) jobs, interleaved so both families make progress."""
    dqn = [(f"t03_{arm}_s{s}", "train_dqn.py", DQN_COMMON + DQN_ARMS[arm] + ["--seed", str(s)], 900) for s in seeds for arm in DQN_ARMS]
    ppo = [(f"t04_ppo_{arm}_s{s}", "train_ppo.py", PPO_COMMON + PPO_ARMS[arm] + ["--seed", str(s)], 900) for s in seeds for arm in PPO_ARMS]
    out = []
    while dqn or ppo:
        for fam in (ppo, dqn):
            for _ in range(2 if fam is dqn else 1):
                if fam:
                    out.append(fam.pop(0))
    return [{"name": n, "script": sc, "args": ar, "peak_mb": pk} for n, sc, ar, pk in out]


def is_done(name):
    return any((RES / d / name / "summary.json").exists() for d in ("phase2_dqn", "phase4_ppo"))


def running_jobs():
    """{job name: pid} of live training processes (the venv launcher and the real python both match; keep one name once)."""
    out = {}
    for p in psutil.process_iter(["pid", "cmdline"]):
        c = p.info["cmdline"] or []
        s = " ".join(c)
        if ("train_dqn.py" in s or "train_ppo.py" in s) and "--name" in c:
            out[c[c.index("--name") + 1]] = p.info["pid"]
    return out


def gpu():
    """(util %, memory used MB, memory total MB); (100, total, total) if nvidia-smi fails (e.g. right after the PC wakes from sleep),
    so the queue waits instead of crashing."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        return int(r[0]), int(r[1]), int(r[2])
    except Exception:
        return 100, 4096, 4096


def latest_rate(name):
    for d in ("phase2_dqn", "phase4_ppo"):
        f = RES / d / name / "metrics.jsonl"
        if f.exists():
            lines = f.read_text().splitlines()
            if lines:
                return json.loads(lines[-1]).get("steps_per_s", 0)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-jobs", type=int, default=3)
    ap.add_argument("--margin-mb", type=int, default=300)
    ap.add_argument("--cpu-limit", type=float, default=92.0)
    ap.add_argument("--util-limit", type=float, default=90.0)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--plan", help="JSON plan file; default: the tryout 03 + 04 plan in this script")
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text()) if a.plan else default_plan()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(ROOT / "python"))
    (RES / "bench").mkdir(parents=True, exist_ok=True)
    tl = RES / "bench" / "jobq_timeline.csv"
    new = not tl.exists()
    f = tl.open("a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["time", "running", "gpu_util_pct", "gpu_mem_mb", "cpu_pct", "aggregate_steps_per_s", "jobs"])
    t0 = time.time()
    queue = [j for j in plan if not is_done(j["name"])]
    print(f"queue: {len(queue)} jobs, max {a.max_jobs} concurrent", flush=True)
    while queue:
        run = running_jobs()
        queue = [j for j in queue if j["name"] not in run and not is_done(j["name"])]
        samples = []
        for _ in range(3):
            u, mem, total = gpu()
            samples.append(u)
            time.sleep(1)
        util = round(sum(samples) / len(samples))
        cpu = psutil.cpu_percent(interval=None)
        agg = sum(latest_rate(n) for n in run)
        w.writerow([round(time.time() - t0), len(run), util, mem, cpu, agg, ";".join(sorted(run))]); f.flush()
        ready = [j for j in queue if all((ROOT / r).exists() for r in j.get("requires", []))]   # e.g. the previous curriculum stage
        if ready and len(run) < a.max_jobs:
            j = ready[0]
            free = total - mem
            if free < j["peak_mb"] + a.margin_mb:
                print(f"[{time.time() - t0:6.0f}s] wait: free VRAM {free} MB < {j['peak_mb']}+{a.margin_mb}", flush=True)
            elif util >= a.util_limit:
                print(f"[{time.time() - t0:6.0f}s] wait: GPU {util}% >= {a.util_limit}% (saturated)", flush=True)
            elif cpu >= a.cpu_limit:
                print(f"[{time.time() - t0:6.0f}s] wait: CPU {cpu:.0f}% >= {a.cpu_limit}%", flush=True)
            else:
                queue.remove(j)
                log = open(RES / f"{j['name']}.log", "wb")
                subprocess.Popen([str(PY), str(ROOT / "python" / j["script"]), "--name", j["name"]] + j["args"], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
                print(f"[{time.time() - t0:6.0f}s] started {j['name']} (running {len(run) + 1}, GPU {util}%, mem {mem}/{total} MB, "
                      f"CPU {cpu:.0f}%)", flush=True)
        time.sleep(a.poll)
    print("queue empty: every job has been launched (running jobs continue on their own)", flush=True)


if __name__ == "__main__":
    main()

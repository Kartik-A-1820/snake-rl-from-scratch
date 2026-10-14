# How many training jobs should share this GPU? Runs k = 1..K identical short jobs at once and records, while they run:
# aggregate env-steps/s (sum over jobs), GPU utilisation, GPU memory in use, CPU utilisation.
# Other training processes already running (matching --pause-match) are suspended during the measurement and resumed
# afterwards, so the numbers are not contaminated by them. Orchestration runs on any Python with psutil; the jobs run in
# .venv-cuda.
#   python python/bench_parallel.py --family ppo --max-jobs 4
#   python python/bench_parallel.py --family dqn --max-jobs 4 --pause-match t03_
# Output: results/bench/parallel_<family>.json
import argparse
import ctypes
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv-cuda" / "Scripts" / "python.exe"
JOBS = {
    # steps are cheap to compare within a family only: a DQN step and a PPO step cost very different amounts of GPU work
    "ppo": (ROOT / "python" / "train_ppo.py", ["--device", "cuda", "--mask", "--total-steps", "700000", "--epoch-steps", "100000",
                                                "--checkpoint-every", "100000"], 100000, 2),
    "dqn": (ROOT / "python" / "train_dqn.py", ["--device", "cuda", "--total-steps", "100000", "--epoch-steps", "20000",
                                                "--learn-start", "10000", "--checkpoint-every", "100000"], 20000, 2),
}
PROCESS_SUSPEND_RESUME = 0x0800


def set_suspended(pids, suspend):
    ntdll, k32 = ctypes.windll.ntdll, ctypes.windll.kernel32
    for pid in pids:
        h = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if h:
            (ntdll.NtSuspendProcess if suspend else ntdll.NtResumeProcess)(h)
            k32.CloseHandle(h)


def find_pids(match):
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if match and match in c and ("train_dqn" in c or "train_ppo" in c):
            out.append(p.info["pid"])
    return out


def gpu():
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().split(",")
    return int(r[0]), int(r[1]), int(r[2])


def run_k(family, k):
    script, args, epoch_steps, skip = JOBS[family]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(ROOT / "python"))
    names = [f"bench_{family}_k{k}_{i}" for i in range(k)]
    procs = [subprocess.Popen([str(PY), str(script), "--name", n, "--seed", str(i)] + args, cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=open(ROOT / "results" / f"{n}.err", "wb")) for i, n in enumerate(names)]
    util, mem, cpu = [], [], []
    psutil.cpu_percent(None)
    t0 = time.time()
    try:
        while any(p.poll() is None for p in procs) and time.time() - t0 < 900:
            time.sleep(2)
            u, m, _ = gpu()
            util.append(u); mem.append(m); cpu.append(psutil.cpu_percent(None))
    finally:
        for p in procs:  # never leave benchmark jobs behind
            if p.poll() is None:
                p.kill()
    sub = "phase4_ppo" if family == "ppo" else "phase2_dqn"
    rates = []
    for n in names:
        f = ROOT / "results" / sub / n / "metrics.jsonl"
        ep = [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []
        use = ep[skip:skip + 2] or ep[-1:]
        rates.append(statistics.mean(e["steps_per_s"] for e in use) if use else 0)
    m0 = ROOT / "results" / sub / names[0] / "metrics.jsonl"
    vram = [json.loads(l)["vram_mb"] for l in m0.read_text().splitlines()] if m0.exists() else []
    vram = vram[-1] if vram else None   # None: the job died (most likely CUDA out of memory next to the suspended jobs)
    # utilisation / memory while all k jobs were certainly alive: the middle half of the samples
    mid = slice(len(util) // 4, 3 * len(util) // 4 or None)
    return {"jobs": k, "per_job_steps_per_s": [round(r) for r in rates], "aggregate_steps_per_s": round(sum(rates)),
            "gpu_util_pct": round(statistics.mean(util[mid] or util)), "gpu_mem_used_mb": round(statistics.mean(mem[mid] or mem)),
            "cpu_util_pct": round(statistics.mean(cpu[mid] or cpu)), "torch_peak_alloc_mb_per_job": vram,
            "wall_s": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(JOBS), required=True)
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--pause-match", default="", help="suspend running train_* processes whose command line contains this")
    a = ap.parse_args()
    paused = find_pids(a.pause_match) if a.pause_match else []
    print(f"suspending {len(paused)} processes: {paused}", flush=True)
    set_suspended(paused, True)
    idle = gpu()
    rows = []
    try:
        for k in range(1, a.max_jobs + 1):
            r = run_k(a.family, k)
            r["gpu_mem_idle_mb"] = idle[1]
            rows.append(r)
            print(json.dumps(r), flush=True)
    finally:
        set_suspended(paused, False)
        print("resumed", flush=True)
    out = ROOT / "results" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"parallel_{a.family}.json").write_text(json.dumps({"family": a.family, "idle_gpu_mem_mb": idle[1],
                                                               "gpu_total_mb": idle[2], "rows": rows}, indent=1))


if __name__ == "__main__":
    main()

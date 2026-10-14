# Writes RESUME.md: what is running, how far it is, and the exact commands to pause and resume. Run it any time.
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RUNS = ROOT / "models" / "_runs"


def running():
    out = {}
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process | ? { $_.Name -match '^python' -and $_.CommandLine -match 'train_(dqn|ppo)' } | % { $_.CommandLine }"],
                       capture_output=True, text=True).stdout
    for line in r.splitlines():
        m = re.search(r"train_(dqn|ppo)\.py.*?--name (\S+)", line)
        if m:
            out[m.group(2)] = line
    return out


def progress(name):
    p = RES / f"{name}.log"
    if not p.exists():
        return None
    t = p.read_text(encoding="utf-8", errors="replace")
    ep = re.findall(r"^ep +\d+ \| steps +([\d.]+)M", t, re.M)
    cp = re.findall(r"CHECKPOINT epoch (\d+): TEST mean ([\d.]+) median \d+ p90 \d+ max (\d+) wins (\d+)", t)
    cfg = next((json.loads(c.read_text(encoding="utf-8")) for c in (RES / "phase2_dqn" / name / "config.json", RES / "phase4_ppo" / name / "config.json") if c.exists()), {})
    total = cfg.get("total_steps")
    return {"steps_m": float(ep[-1]) if ep else 0.0, "total_m": total / 1e6 if total else None, "last_cp": cp[-1] if cp else None}


def main():
    live = running()
    done = sorted(p.parent.name for p in list((RES / "phase2_dqn").glob("*/summary.json")) + list((RES / "phase4_ppo").glob("*/summary.json")))
    L = [f"# RESUME: state written {datetime.now():%Y-%m-%d %H:%M}", "",
         "## Pause safely (power cut, shutdown)",
         "Jobs started after resume support was added stop cleanly: create the file `results/STOP` (every job) or `results/STOP_<run>` (one job).",
         "Each job saves `models/_runs/<run>_state.pt` (network, optimiser, counters, metrics, RNG; not the games in progress, and for DQN not the replay buffer) and exits.",
         "Delete the STOP file, then resume. Sleeping the laptop (not shutting down) is also safe: the processes survive and continue on wake.", "",
         "## Resume",
         "Rerun the same command as the original job with `--resume` added (the job command is in `results/plan_tonight.json`, or see below); then restart the queue:",
         "`python python/jobq.py --max-jobs 5 --plan results/plan_tonight.json` (skips finished and running jobs).", "",
         "Jobs that started *before* resume support existed cannot save on request. If one is lost, restart it from its last evaluation checkpoint: DQN runs refill the replay buffer anyway; PPO stages can start from `models/_runs/<run>.pt` or `_best.pt` with `--init-from`.", "",
         "## Running now"]
    for name, cmd in live.items():
        pr = progress(name)
        has = (RUNS / f"{name}_state.pt").exists()
        L.append(f"- `{name}`: {pr['steps_m']:.2f} M" + (f" of {pr['total_m']:.0f} M ({100 * pr['steps_m'] / pr['total_m']:.0f}%)" if pr and pr["total_m"] else "")
                 + (f"; last checkpoint test mean {pr['last_cp'][1]}, max {pr['last_cp'][2]}, wins {pr['last_cp'][3]}" if pr and pr["last_cp"] else "")
                 + f"; resume state: {'yes' if has else 'no (started before resume support)'}")
        L.append(f"  - command: `{cmd.split('train_')[0].strip().split()[-1] if False else ''}python python/train_{'ppo' if 'train_ppo' in cmd else 'dqn'}.py " + cmd.split(".py", 1)[1].strip() + " --resume`")
    if not live:
        L.append("- (nothing running)")
    L += ["", "## Finished runs", ", ".join(f"`{d}`" for d in done), "", "## Registry", ""]
    for p in sorted((ROOT / "models").glob("*/meta.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        L.append(f"- `{m['id']}`: {m['name']}, test mean {m['test']['mean']:.1f}" + (", provisional" if m.get("provisional") else ""))
    (ROOT / "RESUME.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("RESUME.md written")


if __name__ == "__main__":
    main()

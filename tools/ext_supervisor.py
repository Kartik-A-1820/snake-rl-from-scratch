# Supervises a long, resumable PPO run: keeps it alive, watches the GPU temperature, and backs off the load if the card runs hot.
#   python tools/ext_supervisor.py --name t05_12x12_ext --stop-at 08:00 --total-steps 200000000
# * Launches python/train_ppo.py with --resume (so a crash, a sleep-induced hiccup or a supervisor restart continues from the last saved
#   state, written every logging epoch) and restarts it if it dies without having finished (at most 5 times in a row).
# * Every 30 s it logs temperature, utilisation, SM clock, power and the throttle-reason bits to results/bench/gpu_temp_log.csv.
# * If the temperature stays at or above --temp-limit (default 87 C) for --sustain samples (default 6, i.e. 3 minutes) it logs the event
#   to results/bench/gpu_events.log, asks the job to stop cleanly (results/STOP_<name>), and relaunches it with half the parallel
#   environments (minimum 256), keeping the minibatch size at 2048. It never raises the load again by itself.
# * Status for humans: results/ext_supervisor.json. Stop everything: create results/STOP (the job saves its state and exits).
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PY = ROOT / ".venv-cuda" / "Scripts" / "python.exe"
THERMAL_BITS = {0x20: "SW thermal slowdown", 0x40: "HW thermal slowdown", 0x08: "HW slowdown", 0x80: "HW power brake"}


def gpu():
    q = "temperature.gpu,utilization.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active,memory.used"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        t, u, c, p, th, m = [x.strip() for x in r.strip().splitlines()[0].split(",")]
        return {"temp": int(t), "util": int(u), "sm_mhz": int(c), "power_w": float(p), "throttle": int(th, 16), "mem_mb": int(m)}
    except Exception:
        return None


def command(a, envs):
    mb = envs * 64 // 2048
    c = [str(PY), str(ROOT / "python" / "train_ppo.py"), "--name", a.name, "--device", "cuda", "--size", "12", "--obs", "win6", "--mask",
         "--gamma", "0.995", "--win-reward", "10", "--envs", str(envs), "--minibatches", str(mb), "--horizon", "64",
         "--total-steps", str(a.total_steps), "--epoch-steps", "500000", "--checkpoint-every", "5", "--init-from", str(ROOT / "models" / "_runs" / "init_12x12_ext.pt"),
         "--seed", "3", "--resume", "--plateau-stop", "--plateau-window", str(a.plateau_window), "--plateau-gain", "1.0", "--plateau-win-gain", "2.0",
         "--publish", "05_ppo_win_12x12", "--publish-name", "PPO win challenge, 12x12", "--tryout", "5", "--notes", "../notes/05_win_challenge.html",
         "--description", "PPO with the action mask and a reachable-area channel on the size-agnostic canvas, trained from scratch on the full 12x12 board "
                          "(24 M steps, then an extended run).", "--publish-live"]
    if a.stop_at:
        c += ["--stop-at", a.stop_at]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t05_12x12_ext")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--total-steps", type=int, default=200_000_000)
    ap.add_argument("--plateau-window", type=int, default=25_000_000)
    ap.add_argument("--stop-at", default="08:00")
    ap.add_argument("--temp-limit", type=int, default=87)
    ap.add_argument("--sustain", type=int, default=6)
    ap.add_argument("--poll", type=int, default=30)
    a = ap.parse_args()
    (RES / "bench").mkdir(parents=True, exist_ok=True)
    tl, ev = RES / "bench" / "gpu_temp_log.csv", RES / "bench" / "gpu_events.log"
    new = not tl.exists()
    f = tl.open("a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["time", "temp_c", "util_pct", "sm_mhz", "power_w", "throttle_hex", "mem_mb", "envs"])

    def event(msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with ev.open("a", encoding="utf-8") as e:
            e.write(line + "\n")

    envs, restarts, hot, last_reduce, proc, status = a.envs, 0, 0, 0.0, None, {}
    stop_file = RES / f"STOP_{a.name}"
    log = RES / f"{a.name}.log"
    while True:
        if (RES / "STOP").exists():
            event("results/STOP found: supervisor exits")
            break
        if proc is None:
            stop_file.unlink(missing_ok=True)
            proc = subprocess.Popen(command(a, envs), cwd=ROOT, stdout=log.open("ab"), stderr=subprocess.STDOUT,
                                    env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONPATH": str(ROOT / "python")})
            event(f"launched {a.name} with {envs} envs (restart #{restarts})")
        g = gpu()
        if g:
            w.writerow([datetime.now().isoformat(timespec="seconds"), g["temp"], g["util"], g["sm_mhz"], g["power_w"], hex(g["throttle"]), g["mem_mb"], envs])
            f.flush()
            therm = [n for b, n in THERMAL_BITS.items() if g["throttle"] & b]
            if therm:
                event(f"throttle reason active: {', '.join(therm)} (temp {g['temp']} C, SM {g['sm_mhz']} MHz)")
            hot = hot + 1 if g["temp"] >= a.temp_limit else 0
            status = {"time": datetime.now().isoformat(timespec="seconds"), "envs": envs, "restarts": restarts, **g}
            (RES / "ext_supervisor.json").write_text(json.dumps(status, indent=1))
            if hot >= a.sustain and envs > 256 and time.time() - last_reduce > 1200:
                event(f"temperature {g['temp']} C >= {a.temp_limit} C for {hot} samples: reducing the load from {envs} to {max(256, envs // 2)} envs")
                stop_file.write_text("supervisor: thermal back-off")
                try:
                    proc.wait(timeout=420)
                except subprocess.TimeoutExpired:
                    event("the job did not stop in 7 minutes; terminating it (it resumes from the last saved state)")
                    proc.terminate()
                    proc.wait()
                envs, proc, hot, last_reduce = max(256, envs // 2), None, 0, time.time()
                continue
            if hot >= a.sustain and envs <= 256 and time.time() - last_reduce > 1200:
                event(f"temperature {g['temp']} C still >= {a.temp_limit} C at the minimum load (256 envs): logged, not reducing further")
                last_reduce = time.time()
        if proc is not None and proc.poll() is not None:
            code = proc.returncode
            done = (RES / "phase4_ppo" / a.name / "summary.json").exists()
            if done:
                s = json.loads((RES / "phase4_ppo" / a.name / "summary.json").read_text(encoding="utf-8"))
                event(f"run finished (exit {code}); stop reason: {s.get('stopped') or 'step cap reached'}")
                break
            restarts += 1
            event(f"{a.name} exited with code {code} before finishing")
            if restarts > 5:
                event("too many restarts: giving up")
                break
            proc = None
            time.sleep(60)
            continue
        time.sleep(a.poll)


if __name__ == "__main__":
    main()

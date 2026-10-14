# CPU vs CUDA, measured on this machine instead of assumed:
#   1. inference latency at batch 1 (what the live game needs), for a tiny MLP and the tryout-02 CNN
#   2. one training step (forward + Huber loss + backward + Adam) at batch 64 … 4096
#   3. environment throughput: scalar Python env vs the batched torch env on CPU and CUDA
# Writes results/bench/bench.json (picked up by the tryout-02 notes) and prints tables.
#   python python/bench_device.py [--quick]
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from snake_rl.core import SnakeGame
from snake_rl.device import describe, sync
from snake_rl.nets import make_net, n_params
from snake_rl.vec_env import TorchSnakeVecEnv

ROOT = Path(__file__).resolve().parents[1]
DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])


def timed(fn, dev, warmup, iters, per_call_sync=True):
    for _ in range(warmup):
        fn()
    sync(dev)
    if per_call_sync:  # latency: each call must finish before the next, like a real game loop
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            fn()
            sync(dev)
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)
    t = time.perf_counter()  # throughput: let CUDA queue work, sync once
    for _ in range(iters):
        fn()
    sync(dev)
    return (time.perf_counter() - t) / iters


def inference(quick):
    rows = []
    for name, obs in (("mlp 11→64→64→3", "basic"), ("cnn 4x14x14", "grid")):
        for dev in DEVICES:
            for threads in ([1, torch.get_num_threads()] if dev.type == "cpu" else [None]):
                if threads:
                    torch.set_num_threads(threads)
                net = make_net(obs, 12).to(dev).eval()
                x = torch.rand((1, 11) if obs == "basic" else (1, 4, 14, 14), device=dev)
                with torch.inference_mode():
                    s = timed(lambda: net(x), dev, 50, 300 if quick else 2000)
                rows.append({"model": name, "params": n_params(net), "device": dev.type,
                             "threads": threads, "batch": 1, "latency_us": round(s * 1e6, 1)})
    torch.set_num_threads(max(r["threads"] or 0 for r in rows))
    try:  # ONNX Runtime on CPU: what a deployed agent would typically use
        import onnxruntime as ort
        for name, obs in (("mlp 11→64→64→3", "basic"), ("cnn 4x14x14", "grid")):
            net = make_net(obs, 12).eval()
            x = torch.rand((1, 11) if obs == "basic" else (1, 4, 14, 14))
            path = ROOT / "results" / "bench" / f"_tmp_{obs}.onnx"
            torch.onnx.export(net, (x,), str(path), input_names=["obs"], output_names=["q"])
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            xn = x.numpy()
            s = timed(lambda: sess.run(None, {"obs": xn}), torch.device("cpu"), 50, 300 if quick else 2000)
            rows.append({"model": name, "params": n_params(net), "device": "onnxruntime-cpu", "threads": 1,
                         "batch": 1, "latency_us": round(s * 1e6, 1)})
            path.unlink(missing_ok=True)
    except Exception as e:
        print(f"  (onnxruntime row skipped: {e})")
    return rows


def train_step(quick):
    rows = []
    for name, obs in (("mlp 11→64→64→3", "basic"), ("cnn 4x14x14", "grid")):
        for dev in DEVICES:
            for b in (64, 256, 1024, 4096):
                net = make_net(obs, 12).to(dev)
                opt = torch.optim.Adam(net.parameters(), lr=1e-4)
                x = torch.rand((b, 11) if obs == "basic" else (b, 4, 14, 14), device=dev)
                a = torch.randint(0, 3, (b,), device=dev)
                y = torch.rand(b, device=dev)

                def step():
                    q = net(x).gather(1, a.unsqueeze(1)).squeeze(1)
                    loss = F.smooth_l1_loss(q, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                iters = 20 if (quick or (dev.type == "cpu" and obs == "grid" and b >= 1024)) else 100
                s = timed(step, dev, 5, iters, per_call_sync=False)
                rows.append({"model": name, "device": dev.type, "batch": b, "ms_per_step": round(s * 1e3, 3),
                             "samples_per_s": round(b / s)})
    return rows


def env_throughput(quick):
    rows = []
    g = SnakeGame(12, 12, seed=0)
    n, t = 0, time.perf_counter()
    while time.perf_counter() - t < (1 if quick else 3):
        g.step(n % 3 if n % 7 else 0)
        n += 1
        if g.done:
            g.reset()
    rows.append({"env": "scalar python", "device": "cpu", "n_envs": 1, "steps_per_s": round(n / (time.perf_counter() - t))})
    for dev in DEVICES:
        for n_envs in (1, 64, 256, 1024, 4096):
            env = TorchSnakeVecEnv(n_envs, 12, device=dev)
            acts = torch.randint(0, 3, (64, n_envs), device=dev)
            k = 0

            def step():
                nonlocal k
                env.step(acts[k % 64])
                k += 1

            iters = 30 if quick else 200
            s = timed(step, dev, 5, iters, per_call_sync=False)
            rows.append({"env": "torch vec (grid obs)", "device": dev.type, "n_envs": n_envs,
                         "steps_per_s": round(n_envs / s), "ms_per_step": round(s * 1e3, 3)})
    return rows


def table(rows, cols):
    w = {c: max(len(c), *(len(str(r.get(c))) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  " + "  ".join(str(r.get(c)).ljust(w[c]) for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    (ROOT / "results" / "bench").mkdir(parents=True, exist_ok=True)
    print("devices:", ", ".join(describe(d) for d in DEVICES))
    res = {"machine": {"platform": platform.platform(), "devices": [describe(d) for d in DEVICES]}}
    print("\n1) inference latency, batch 1 (median)")
    res["inference"] = inference(a.quick)
    table(res["inference"], ["model", "params", "device", "threads", "latency_us"])
    print("\n2) one training step (fwd + Huber + bwd + Adam)")
    res["train_step"] = train_step(a.quick)
    table(res["train_step"], ["model", "device", "batch", "ms_per_step", "samples_per_s"])
    print("\n3) environment throughput (random actions)")
    res["env"] = env_throughput(a.quick)
    table(res["env"], ["env", "device", "n_envs", "steps_per_s"])
    (ROOT / "results" / "bench" / "bench.json").write_text(json.dumps(res, indent=1))
    print("\nwrote results/bench/bench.json")

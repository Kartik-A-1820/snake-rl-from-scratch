# Export a DQN checkpoint: game-format JSON (+ test vectors), ONNX (+ parity and latency check), and publish it
# into the model registry (models/<algo_id>/, see registry.py). Used by python/export_model.py and, with --publish,
# by python/train_dqn.py right after a run, so the game's Agent dropdown is always current.
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import registry
from .core import SnakeGame, greedy_action
from .features import basic_features, grid_obs, win_obs
from .nets import ActorCritic, make_net, n_params

ROOT = registry.ROOT


def layers_of(seq):
    out = []
    for m in seq:
        if isinstance(m, nn.Conv2d):
            out.append({"op": "conv", "cin": m.in_channels, "cout": m.out_channels, "k": m.kernel_size[0],
                        "stride": m.stride[0], "pad": m.padding[0], "w": rnd(m.weight), "b": rnd(m.bias)})
        elif isinstance(m, nn.Linear):
            out.append({"op": "linear", "in": m.in_features, "out": m.out_features, "w": rnd(m.weight), "b": rnd(m.bias)})
        elif isinstance(m, nn.ReLU):
            out.append({"op": "relu"})
        elif isinstance(m, nn.Flatten):
            out.append({"op": "flatten"})
        else:
            raise TypeError(f"no JS op for {type(m).__name__}")
    return out


def rnd(t):
    # 6 significant digits: ~3x smaller JSON, error far below anything that flips an argmax in practice
    return [float(f"{v:.6g}") for v in t.detach().cpu().flatten().tolist()]


def sample_states(obs, size, n=8):
    # real positions from greedy play (short and long snakes), not random noise
    rng, xs, g = random.Random(0), [], SnakeGame(size, size, seed=4242)
    while len(xs) < n:
        g.step(greedy_action(g) if rng.random() < 0.9 else rng.randrange(3))
        if g.done:
            g.reset()
        if rng.random() < 0.05:
            xs.append(grid_obs(g) if obs == "grid" else win_obs(g, obs == "win6") if obs in ("win5", "win6")
                      else np.asarray(basic_features(g), dtype=np.float32))
    return np.stack(xs)


def onnx_export(net, x, path):
    try:
        torch.onnx.export(net, (x,), str(path), input_names=["obs"], output_names=["q"],
                          dynamic_axes={"obs": {0: "batch"}, "q": {0: "batch"}})
    except Exception as e:  # newer torch defaults to the dynamo exporter; fall back to the classic one
        print(f"  default ONNX exporter failed ({e.__class__.__name__}); retrying with dynamo=False")
        torch.onnx.export(net, (x,), str(path), input_names=["obs"], output_names=["q"],
                          dynamic_axes={"obs": {0: "batch"}, "q": {0: "batch"}}, dynamo=False)


def embed_onnx(path):
    # the dynamo exporter may write weights to <file>.data; fold them back so best.onnx is one self-contained file
    import onnx
    m = onnx.load(str(path))
    onnx.save_model(m, str(path), save_as_external_data=False)
    side = Path(str(path) + ".data")
    if side.exists():
        side.unlink()


def clean_epochs(epochs):
    """Copy of the per-epoch log with epochs inflated by the computer sleeping (wall time far above the run's median epoch, e.g. a
    12 h gap overnight) replaced by the median epoch, so training time and throughput describe the work, not the sleep."""
    sec = sorted(x["seconds"] for x in epochs)
    med = sec[len(sec) // 2] if sec else 0
    out = []
    for x in epochs:
        if med and x["seconds"] > max(8 * med, 900):
            x = dict(x, seconds=med, steps_per_s=round(x.get("env_steps_per_epoch", 0) or 0) or round(statistics.median(y["steps_per_s"] for y in epochs)))
        out.append(x)
    return out


def run_cost(run_name):
    """Env steps / wall seconds / throughput / VRAM of a training run, from its summary.json (None if missing)."""
    p = next((q for q in (ROOT / "results" / d / run_name / "summary.json" for d in ("phase2_dqn", "phase4_ppo")) if q.exists()), None)
    if p is None:
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    e = clean_epochs(s["epochs"])
    cfg = s["config"]
    return {"run": run_name, "env_steps": e[-1]["env_steps"], "updates": e[-1]["updates"],
            "seconds": round(sum(x["seconds"] for x in e)), "steps_per_s": round(statistics.mean(x["steps_per_s"] for x in e)),
            "vram_mb": e[-1].get("vram_mb"), "device": cfg.get("device"), "batch": cfg.get("batch"),
            "n_step": cfg.get("n_step", 1), "dueling": cfg.get("dueling", False), "double": not cfg.get("no_double", False),
            "algo": s.get("tryout")}


def run_replays(run_name):
    """Recorded winning games of a run (first wins on the held-out seeds), for the game's "Replay a win" button."""
    live = ROOT / "results" / "phase4_ppo" / run_name / "best_replays.json"
    if live.exists():   # winning games of the current best checkpoint (written during the run)
        return json.loads(live.read_text(encoding="utf-8"))[:3]
    p = next((q for q in (ROOT / "results" / d / run_name / "summary.json" for d in ("phase2_dqn", "phase4_ppo")) if q.exists()), None)
    if p is None:
        return []
    fw = json.loads(p.read_text(encoding="utf-8")).get("first_win")
    return (fw or {}).get("replays", [])


def run_curve(run_name):
    """[[env_steps, greedy test mean], ...] at every checkpoint of a run, for the game's mini learning curve."""
    p = next((q for q in (ROOT / "results" / d / run_name / "summary.json" for d in ("phase2_dqn", "phase4_ppo")) if q.exists()), None)
    if p is None:
        return None
    return [[c["env_steps"], round(c["test"]["mean"], 2)] for c in json.loads(p.read_text(encoding="utf-8"))["checkpoints"]]


@torch.no_grad()
def evaluate_exported(net, c, seeds):
    """Greedy evaluation of the exported net on the held-out seeds (honours the action mask); returns mean/std of the scores."""
    from .vec_env import evaluate
    dev = "cuda" if torch.cuda.is_available() else "cpu"   # the 500-game evaluation is far too slow on the CPU for the larger boards
    net = net.to(dev)
    from .nets import ActorCritic  # noqa: F401  (forward() of every net type returns Q-values / logits)
    if c.get("mask"):
        def pol(x, d):  # d: (n, 3) bool, moves that kill at once; if every move kills, none is masked
            ok = ~d
            ok = ok | ~ok.any(1, keepdim=True)
            return net(x).masked_fill(~ok, float("-inf")).argmax(1)
        r = evaluate(pol, seeds, size=c["size"], device=dev, obs=c["obs"], masked=True, trap=bool(c.get("trap_mask")))
    else:
        r = evaluate(lambda x: net(x).argmax(1), seeds, size=c["size"], device=dev, obs=c["obs"])
    sc = np.asarray(r["scores"], dtype=np.float64)
    return {"mean": float(sc.mean()), "std": float(sc.std()), "max": int(sc.max())}


def export_checkpoint(ckpt, algo_id=None, name=None, tryout=None, notes=None, description="", run=None,
                      only_if_better=False, extra=None, provisional=False):
    """Export `ckpt` (a *_best.pt). With algo_id, publish to models/<algo_id>/ and rebuild registry.js.
    only_if_better: skip publishing unless this checkpoint's validation mean beats the registered one."""
    ck = torch.load(ckpt, map_location="cpu")
    c = ck["config"]
    if algo_id and only_if_better and not registry.should_replace(algo_id, ck.get("val_mean")):
        print(f"[registry] {algo_id}: the registered model has a higher validation mean; not replaced")
        return None
    net = (ActorCritic(c.get("net_size", c["size"]), in_ch=c.get("in_ch", 4)) if c.get("algo") == "ppo"
           else make_net(c["obs"], c["size"], c["dueling"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    stem = Path(ckpt).stem
    layers = layers_of(net.body)
    if net.dueling:
        layers.append({"op": "dueling", "value": layers_of(net.value), "adv": layers_of(net.adv)})
    else:
        layers += layers_of(net.head)
    xs = sample_states(c["obs"], c["size"])
    with torch.no_grad():
        qs = net(torch.from_numpy(xs)).numpy()
    model = {"type": "net", "obs": "grid4" if c["obs"] == "grid" else c["obs"] if c["obs"] in ("win5", "win6") else None, "features": "basic" if c["obs"] == "basic" else None,
             "size": c["size"], "mask": ("trap" if c.get("trap_mask") else bool(c.get("mask", False))), "params": n_params(net), "epoch": ck.get("epoch"), "test": ck.get("test"), "layers": layers,
             "tests": [{"x": [float(f"{v:.6g}") for v in x.flatten()], "shape": list(x.shape), "q": q.tolist()}
                       for x, q in zip(xs, qs)]}

    out_dir = (registry.MODELS / algo_id) if algo_id else (registry.MODELS / "_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    opath = out_dir / ("best.onnx" if algo_id else f"{stem}.onnx")
    onnx_export(net, torch.from_numpy(xs[:1]), opath)
    embed_onnx(opath)
    report = {"name": stem, "params": n_params(net), "fp32_mb": round(n_params(net) * 4 / 1e6, 3),
              "onnx_mb": round(opath.stat().st_size / 1e6, 3)}
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(opath), so, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"obs": xs})[0]
        report["onnx_max_abs_err"] = float(np.abs(out - qs).max())
        ts = []
        for _ in range(1000):
            t = time.perf_counter()
            sess.run(None, {"obs": xs[:1]})
            ts.append(time.perf_counter() - t)
        report["onnx_cpu_latency_us"] = round(statistics.median(ts[100:]) * 1e6, 1)
    except ImportError:
        report["onnx"] = "onnxruntime not installed"

    if not algo_id:  # flat export next to the checkpoint
        (registry.MODELS / "_runs" / f"{stem}.json").write_text(json.dumps(model, separators=(",", ":")))
        (ROOT / "results" / "bench").mkdir(parents=True, exist_ok=True)
        (ROOT / "results" / "bench" / f"export_{stem}.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return report

    test = dict(ck.get("test") or {})
    ev = evaluate_exported(net, c, registry.TEST_SEEDS)
    test["std"] = ev["std"]
    if test.get("mean") is not None and abs(ev["mean"] - test["mean"]) > 1e-3:
        print(f"[registry] WARNING: re-evaluated test mean {ev['mean']:.3f} differs from the checkpoint's {test['mean']:.3f}")
    test["mean"] = ev["mean"]
    run_name = run or stem.removesuffix("_best")
    cost = run_cost(run_name)
    meta = {"name": name or algo_id, "tryout": tryout, "family": "ppo" if c.get("algo") == "ppo" else "dqn", "description": description,
            "obs": c["obs"], "size": c["size"], "size_locked": c["obs"] == "grid", "mask": ("trap" if c.get("trap_mask") else bool(c.get("mask", False))),
            "components": [k for k, on in (("double", c.get("double", True) and c.get("algo") != "ppo"), ("dueling", c["dueling"]),
                                           ("per", c.get("per", False)), ("mask", c.get("mask", False)), ("trap shield", c.get("trap_mask", False))) if on],
            "params": n_params(net), "fp32_mb": report["fp32_mb"], "onnx_mb": report["onnx_mb"],
            "onnx_cpu_latency_us": report.get("onnx_cpu_latency_us"), "onnx_max_abs_err": report.get("onnx_max_abs_err"),
            "epoch": ck.get("epoch"), "select_score": ck.get("val_mean"), "val_mean": ck.get("val_mean"),
            "test": {k: test.get(k) for k in ("n", "mean", "std", "median", "p10", "p90", "max", "wins", "causes")},
            "requires": ("board image 4x14x14 (grid4)" if c["obs"] == "grid" else "14x14 canvas: tail order, head, food, wall, length"
                         + (", reachable area" if c["obs"] == "win6" else "") if c["obs"] in ("grid", "win5", "win6") else "11 hand-made features"),
            "replays": run_replays(run_name),
            "train": cost, "curve": run_curve(run_name), "notes": notes, "source_ckpt": Path(ckpt).name}
    meta["provisional"] = bool(provisional)   # a best-so-far snapshot of a run that is still training
    meta.update(extra or {})
    d = registry.publish(algo_id, model, meta, pt_src=ckpt)
    print(f"[registry] published {algo_id} -> {d}")
    return meta

# Where does a PPO iteration spend its time on this GPU, and which levers raise useful throughput?
# Mimics python/train_ppo.py (12x12 board, canvas obs with reachable area, instant-death mask) in a loop of full iterations
# (rollout of T steps in N envs, GAE, K epochs of minibatch updates) and reports env steps/s, GPU utilisation, peak memory and,
# with --split, the time per phase (env step, policy forward, sampling/masking, GAE, update), measured with a sync after each phase.
#   python python/profile_ppo.py --split                         # time split of the baseline (N=256)
#   python python/profile_ppo.py --envs 512 1024 --amp --channels-last
# Output: results/bench/ppo_levers.json (appended per configuration).
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from snake_rl.nets import ActorCritic
from snake_rl.vec_env import TorchSnakeVecEnv
from train_ppo import masked_logits, safe_moves

ROOT = Path(__file__).resolve().parents[1]


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.util, self.stop = [], False

    def run(self):
        while not self.stop:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                self.util.append(int(r.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.4)


def run_config(N, T, minibatches, epochs, amp, channels_last, iters, split, size=12, lazy=True, cudnn=True):
    torch.backends.cudnn.benchmark = cudnn
    dev = torch.device("cuda")
    env = TorchSnakeVecEnv(N, size, seeds=list(range(N)), device=dev, obs="win6", gamma=0.995, win_reward=10.0)
    net = ActorCritic(12, in_ch=6).to(dev)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(net.parameters(), lr=2.5e-4, eps=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    obs = env.obs()
    batch = N * T
    mb = batch // minibatches
    obs_b = torch.zeros((T, N, *obs.shape[1:]), device=dev)
    act_b = torch.zeros((T, N), dtype=torch.long, device=dev)
    logp_b, val_b, rew_b, done_b = (torch.zeros((T, N), device=dev) for _ in range(4))
    ok_b = torch.ones((T, N, 3), dtype=torch.bool, device=dev)
    tm = {k: 0.0 for k in ("env", "forward", "sample", "gae", "update")}

    def tick(key, t0):
        if split:
            torch.cuda.synchronize()
            tm[key] += time.perf_counter() - t0
        return time.perf_counter()

    def to_in(x):
        return x.contiguous(memory_format=torch.channels_last) if channels_last else x

    def iteration():
        nonlocal obs
        with torch.no_grad():
            for t in range(T):
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits, v = net.forward_ac(to_in(obs))
                logits, v = logits.float(), v.float()
                t0 = tick("forward", t0)
                ok = safe_moves(env.danger())
                lp = F.log_softmax(masked_logits(logits, ok), dim=-1)
                act = torch.multinomial(lp.exp(), 1).squeeze(1)
                obs_b[t], act_b[t], val_b[t], ok_b[t] = obs, act, v, ok
                logp_b[t] = lp.gather(1, act.unsqueeze(1)).squeeze(1)
                t0 = tick("sample", t0)
                nxt, rew, term, trunc, info = env.step(act, lazy_next=lazy) if lazy else env.step(act)
                tr = trunc & ~term
                if (info["trunc_any"] if lazy else bool(tr.any())):
                    rew = rew + 0.995 * net.value(to_in(info["next_obs"])) * tr.float()
                rew_b[t], done_b[t] = rew, (term | trunc).float()
                obs = nxt
                t0 = tick("env", t0)
            t0 = time.perf_counter()
            last_v = net.value(to_in(obs)).float()
            adv = torch.zeros_like(rew_b)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else val_b[t + 1]
                nonterm = 1.0 - done_b[t]
                gae = rew_b[t] + 0.995 * nv * nonterm - val_b[t] + 0.995 * 0.95 * nonterm * gae
                adv[t] = gae
            ret = adv + val_b
            tick("gae", t0)
        t0 = time.perf_counter()
        b_obs, b_act = obs_b.reshape(batch, *obs.shape[1:]), act_b.reshape(batch)
        b_logp, b_val, b_adv, b_ret, b_ok = logp_b.reshape(batch), val_b.reshape(batch), adv.reshape(batch), ret.reshape(batch), ok_b.reshape(batch, 3)
        for _ in range(epochs):
            perm = torch.randperm(batch, device=dev)
            for i in range(minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits, v = net.forward_ac(to_in(b_obs[idx]))
                logits, v = logits.float(), v.float()
                lpa = F.log_softmax(masked_logits(logits, b_ok[idx]), dim=-1)
                logp = lpa.gather(1, b_act[idx].unsqueeze(1)).squeeze(1)
                p = lpa.exp()
                ent = -(p * lpa.masked_fill(p == 0, 0.0)).sum(-1).mean()
                ratio = (logp - b_logp[idx]).exp()
                madv = b_adv[idx]
                madv = (madv - madv.mean()) / (madv.std() + 1e-8)
                pg = torch.max(-madv * ratio, -madv * ratio.clamp(0.8, 1.2)).mean()
                vc = b_val[idx] + (v - b_val[idx]).clamp(-0.2, 0.2)
                vl = 0.5 * torch.max((v - b_ret[idx]).pow(2), (vc - b_ret[idx]).pow(2)).mean()
                loss = pg + 0.5 * vl - 0.01 * ent
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                scaler.step(opt)
                scaler.update()
        tick("update", t0)
        return float(loss)

    iteration()                      # warm-up (cudnn autotune, allocator)
    torch.cuda.synchronize()
    for k in tm:
        tm[k] = 0.0
    torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler()
    sampler.start()
    t0 = time.perf_counter()
    for _ in range(iters):
        last_loss = iteration()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    sampler.stop = True
    res = {"envs": N, "horizon": T, "minibatches": minibatches, "minibatch_size": mb, "epochs": epochs, "amp": amp, "channels_last": channels_last,
           "lazy_next_obs": lazy, "cudnn_benchmark": cudnn, "steps_per_s": round(N * T * iters / wall), "gpu_util_pct": round(sum(sampler.util) / max(1, len(sampler.util))),
           "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20), "last_loss": round(last_loss, 4)}
    if split:
        tot = sum(tm.values())
        res["time_split_pct"] = {k: round(100 * v / tot, 1) for k, v in tm.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[256])
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--no-lazy", action="store_true")
    ap.add_argument("--no-cudnn-benchmark", action="store_true")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = ROOT / "results" / "bench" / "ppo_levers.json"
    rows = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    for N in a.envs:
        r = run_config(N, a.horizon, a.minibatches, a.epochs, a.amp, a.channels_last, a.iters, a.split, lazy=not a.no_lazy, cudnn=not a.no_cudnn_benchmark)
        r["tag"] = a.tag
        print(json.dumps(r), flush=True)
        rows.append(r)
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

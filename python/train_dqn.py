# Tryout 02: DQN with a state that can see the body.
#   obs "grid": 4x14x14 board image (body recency, head, food, walls) → small CNN   (the point of this tryout)
#   obs "basic": the 11 features from tryout 01 → small MLP                       (ablation: same algorithm, blind state)
# Mnih et al. 2015 DQN = Q-learning + neural net + experience replay + target network. Plus Double-DQN targets
# (van Hasselt et al. 2016) because they cost one line. Envs run batched on the same device as the net.
#
#   python python/train_dqn.py                         # defaults: grid/CNN, 12x12, 10M steps, CUDA if available
#   python python/train_dqn.py --total-steps 1000000   # quick smoke run (~minutes)
#   python python/train_dqn.py --obs basic             # ablation
#
# Logs: one line per epoch (50k env steps); a full greedy evaluation every 25 epochs on 500 held-out seeds.
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from snake_rl.core import SnakeGame
from snake_rl.device import describe, pick_device, sync
from snake_rl.features import basic_features, grid_obs
from snake_rl.nets import make_net, n_params
from snake_rl.resume import load_state, save_state, state_path, stop_requested
from snake_rl.vec_env import CAUSES, TorchSnakeVecEnv, evaluate

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models" / "_runs"   # raw checkpoints of every run; the best per algorithm is published to models/<algo>/
TEST_SEEDS = list(range(100000, 100500))   # same held-out seeds as tryout 01, so numbers compare directly
VAL_SEEDS = list(range(200000, 200200))    # used to pick the "best" checkpoint; never the test seeds


class Replay:
    # Transitions live on the training device as uint8 (obs values are k/255-exact for k ≤ 255), so
    # sampling never crosses PCIe. 200k transitions of a 4x14x14 grid ≈ 314 MB.
    # per=True: prioritized replay (Schaul et al. 2016, proportional variant). prio holds p_i^alpha; new transitions get
    # the current maximum so each is replayed at least once; sampling is multinomial over prio, and the importance
    # weights w_i = (N * P(i))^-beta / max w undo the bias this sampling introduces.
    def __init__(self, cap, obs_shape, device, per=False, alpha=0.6, eps=1e-6):
        self.cap, self.size, self.i, self.device = cap, 0, 0, device
        self.per, self.alpha, self.eps = per, alpha, eps
        self.obs = torch.zeros((cap, *obs_shape), dtype=torch.uint8, device=device)
        self.nxt = torch.zeros((cap, *obs_shape), dtype=torch.uint8, device=device)
        self.act = torch.zeros(cap, dtype=torch.long, device=device)
        self.rew = torch.zeros(cap, device=device)
        self.disc = torch.zeros(cap, device=device)   # gamma^k * (1 - terminal): multiplies the bootstrap value
        self.nmask = torch.zeros((cap, 3), dtype=torch.bool, device=device)  # instant-death moves in the next state
        self.prio = torch.zeros(cap, device=device)
        self.max_prio = torch.ones((), device=device)

    def add(self, o, a, r, o2, disc, nmask):
        n = o.shape[0]
        idx = (torch.arange(n, device=self.device) + self.i) % self.cap
        self.obs[idx], self.nxt[idx] = (o * 255).round().byte(), (o2 * 255).round().byte()
        self.act[idx], self.rew[idx], self.disc[idx], self.nmask[idx] = a, r, disc, nmask
        if self.per:
            self.prio[idx] = self.max_prio
        self.i = (self.i + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, b, beta=0.4):
        if self.per:
            p = self.prio[:self.size]
            idx = torch.multinomial(p, b, replacement=True)
            w = (self.size * p[idx] / p.sum()).pow(-beta)
            w = w / w.max()
        else:
            idx, w = torch.randint(0, self.size, (b,), device=self.device), None
        batch = (self.obs[idx].float() / 255, self.act[idx], self.rew[idx], self.nxt[idx].float() / 255, self.disc[idx],
                 self.nmask[idx])
        return batch, idx, w

    def update_priorities(self, idx, td_abs):
        if self.per:
            p = (td_abs + self.eps).pow(self.alpha)
            self.prio[idx] = p
            self.max_prio = torch.maximum(self.max_prio, p.max())

    def nbytes(self):
        return sum(t.element_size() * t.numel() for t in (self.obs, self.nxt, self.act, self.rew, self.disc, self.nmask, self.prio))


def safe_moves(danger):
    # moves that do not kill on the next tick; if every move kills, leave them all open (the game is lost anyway)
    ok = ~danger
    return ok | ~ok.any(1, keepdim=True)


def td_loss(net, target, batch, weights, double, mask):
    o, a, r, o2, disc, nmask = batch
    q = net(o).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        ok = safe_moves(nmask) if mask else None
        if double:
            # online net picks the action, target net scores it: removes the max-operator's upward bias
            q_on = net(o2)
            a2 = (q_on.masked_fill(~ok, float("-inf")) if mask else q_on).argmax(1, keepdim=True)
            q2 = target(o2).gather(1, a2).squeeze(1)
        else:
            q_t = target(o2)
            q2 = (q_t.masked_fill(~ok, float("-inf")) if mask else q_t).max(1).values
        # disc = gamma^k * (1 - terminal): no bootstrap past a real terminal, while a truncated (timeout)
        # transition still bootstraps from its next state. k is the n-step horizon actually used.
        y = r + disc * q2
    per_sample = F.smooth_l1_loss(q, y, reduction="none")
    loss = per_sample.mean() if weights is None else (weights * per_sample).mean()  # PER: importance-sampling correction
    return loss, q.mean(), (q - y).detach().abs()


class NStep:
    # Turns 1-step vector-env transitions into n-step ones: R = r_t + g*r_{t+1} + ... up to the episode end, then
    # bootstrap from the state after the last step taken. Per-env windows; episode ends cut the window short.
    def __init__(self, n, gamma):
        self.n, self.gamma, self.q = n, gamma, []

    def push(self, o, a, r, o2, term, done, nm):
        self.q.append((o, a, r, o2, term, done, nm))
        if len(self.q) < self.n:
            return None
        o0, a0 = self.q[0][0], self.q[0][1]
        R, alive = torch.zeros_like(r), torch.ones_like(r, dtype=torch.bool)
        end_o, disc, end_m = self.q[0][3], torch.ones_like(r), self.q[0][6]
        for j, (_, _, rj, oj2, tj, dj, mj) in enumerate(self.q):
            R = R + alive * (self.gamma ** j) * rj
            end_o = torch.where(alive.view(-1, *[1] * (o2.dim() - 1)), oj2, end_o)
            end_m = torch.where(alive.unsqueeze(1), mj, end_m)
            disc = torch.where(alive, torch.full_like(disc, self.gamma ** (j + 1)), disc)
            disc = torch.where(alive & tj, torch.zeros_like(disc), disc)
            alive = alive & ~dj
        self.q.pop(0)
        return o0, a0, R, end_o, disc, end_m


def greedy_eval(net, seeds, size, device, obs, mask=False):
    net.eval()
    if mask:
        res = evaluate(lambda x, d: net(x).masked_fill(~safe_moves(d), float("-inf")).argmax(1), seeds, size=size,
                       device=device, obs=obs, masked=True)
    else:
        res = evaluate(lambda x: net(x).argmax(1), seeds, size=size, device=device, obs=obs)
    net.train()
    return res


@torch.no_grad()
def record_replays(net, size, seeds, device, obs, mask=False):
    # Scalar-engine replays (seed + action string + Q-values) for the notes and the browser game.
    net.eval()
    out = []
    for seed in seeds:
        g, acts, qs = SnakeGame(size, size, seed=seed), [], []
        while not g.done:
            x = grid_obs(g) if obs == "grid" else basic_features(g)
            q = net(torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0))[0].cpu().tolist()
            bad = basic_features(g)[:3] if mask else [0, 0, 0]  # danger bits: straight, right, left
            if all(bad):
                bad = [0, 0, 0]
            a = max((i for i in range(3) if not bad[i]), key=lambda i: (q[i], -i))  # first max, like the JS argmax
            acts.append(a)
            qs.append([round(v, 3) for v in q])
            g.step(a)
        out.append({"seed": seed, "width": size, "height": size, "actions": "".join(map(str, acts)),
                    "score": g.score, "cause": g.cause, "q": qs})
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", choices=["grid", "basic"], default="grid")
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="continue from models/_runs/<name>_state.pt (replay buffer is not saved and refills)")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--total-steps", type=int, default=10_000_000)
    ap.add_argument("--epoch-steps", type=int, default=50_000)
    ap.add_argument("--checkpoint-every", type=int, default=25, help="epochs between full evaluations")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--updates-per-iter", type=int, default=4, help="gradient steps per vectorised env step")
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--learn-start", type=int, default=20_000)
    ap.add_argument("--target-every", type=int, default=500, help="gradient steps between target-net syncs")
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.02)
    ap.add_argument("--eps-frac", type=float, default=0.4, help="fraction of training over which ε decays")
    ap.add_argument("--n-step", type=int, default=3, help="n-step return horizon (1 = plain DQN)")
    ap.add_argument("--shaping", choices=["none", "naive", "potential"], default="none",
                    help="reward shaping toward food: naive = +-k per move, potential = gamma*phi(s')-phi(s) (policy-invariant)")
    ap.add_argument("--shape-k", type=float, default=1.0, help="shaping scale (naive: reward per move; potential: phi range [-k, 0])")
    ap.add_argument("--shape-anneal", type=float, default=0.0, help="linearly anneal k to 0 over this fraction of training (0 = off)")
    ap.add_argument("--timeout-penalty", type=float, default=0.0, help="extra reward on a starvation timeout (e.g. -1 style: pass 1.0)")
    ap.add_argument("--publish", metavar="ALGO_ID", help="after training, publish the best checkpoint to models/<ALGO_ID>/ "
                    "(replaces the registered one only if its validation mean is higher) and rebuild models/registry.js")
    ap.add_argument("--publish-name", help="display name in the game's Agent dropdown")
    ap.add_argument("--tryout", type=int, help="tryout number shown in the Agent dropdown")
    ap.add_argument("--notes", help="note link relative to game/index.html, e.g. ../notes/03_dqn_upgrades.html")
    ap.add_argument("--description", default="")
    ap.add_argument("--per", action="store_true", help="prioritized experience replay (Schaul et al. 2016)")
    ap.add_argument("--per-alpha", type=float, default=0.6, help="priority exponent (0 = uniform)")
    ap.add_argument("--per-beta0", type=float, default=0.4, help="importance-sampling exponent at the start; anneals to 1")
    ap.add_argument("--mask", action="store_true", help="action masking: never choose, or bootstrap through, a move that kills at once")
    ap.add_argument("--no-double", action="store_true")
    ap.add_argument("--dueling", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    name = a.name or f"dqn_{'cnn' if a.obs == 'grid' else 'mlp'}_{a.size}x{a.size}"
    run_dir = ROOT / "results" / "phase2_dqn" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    env = TorchSnakeVecEnv(a.envs, a.size, seeds=[10_000_000 * (a.seed + 1) + i for i in range(a.envs)], device=dev, obs=a.obs,
                           shaping=a.shaping, shape_k=a.shape_k, gamma=a.gamma, timeout_penalty=a.timeout_penalty)
    net = make_net(a.obs, a.size, a.dueling).to(dev)
    target = make_net(a.obs, a.size, a.dueling).to(dev)
    target.load_state_dict(net.state_dict())
    target.eval()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    obs = env.obs()
    nstep = NStep(a.n_step, a.gamma)
    buf = Replay(a.buffer, tuple(obs.shape[1:]), dev, per=a.per, alpha=a.per_alpha)
    cfg = {k: v for k, v in vars(a).items()}
    cfg.update(device=describe(dev), params=n_params(net), model_bytes_fp32=n_params(net) * 4, replay_bytes=buf.nbytes())
    print(f"[model] {name}: {n_params(net):,} params ({n_params(net) * 4 / 1e6:.2f} MB fp32) · replay {buf.nbytes() / 1e6:.0f} MB on {dev}")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))

    def stats0():
        z = lambda: torch.zeros((), device=dev)
        return {"eps_n": z(), "score_sum": z(), "score_max": z(), "wins": z(), "causes": torch.zeros(5, device=dev),
                "loss_sum": z(), "q_sum": z(), "updates": 0}

    st, metrics, checkpoints = stats0(), [], []
    steps = updates = 0
    best_val = -1.0
    t_epoch = time.time()
    n_iters = a.total_steps // a.envs
    it0 = 0
    saved = load_state(name, dev) if a.resume else None
    if saved:
        net.load_state_dict(saved["net"])
        target.load_state_dict(saved["target"])
        opt.load_state_dict(saved["opt"])
        steps, updates, it0, best_val = saved["steps"], saved["updates"], saved["it"], saved["best_val"]
        metrics, checkpoints = saved["metrics"], saved["checkpoints"]
        obs = env.obs()
        print(f"[resume] {name}: iteration {it0}/{n_iters}, {steps / 1e6:.2f} M steps; the replay buffer starts empty and refills", flush=True)
    elif a.resume:
        print(f"[resume] no saved state for {name}: starting from the beginning", flush=True)
    log = open(run_dir / "metrics.jsonl", "a" if saved else "w")

    def snapshot(next_it):
        return {"net": net.state_dict(), "target": target.state_dict(), "opt": opt.state_dict(), "steps": steps, "updates": updates,
                "it": next_it, "best_val": best_val, "metrics": metrics, "checkpoints": checkpoints}

    for it in range(it0, n_iters):
        if stop_requested(name):
            save_state(name, snapshot(it))
            log.close()
            print(f"[stop] stop file found: state saved to {state_path(name)} at iteration {it} ({steps / 1e6:.2f} M steps). "
                  f"Resume with --resume.", flush=True)
            return
        eps = max(a.eps_end, a.eps_start - (a.eps_start - a.eps_end) * steps / (a.eps_frac * a.total_steps))
        if a.shape_anneal > 0:
            env.shape_k = a.shape_k * max(0.0, 1.0 - steps / (a.shape_anneal * a.total_steps))
        with torch.no_grad():
            q_now = net(obs)
            if a.mask:
                ok = safe_moves(env.danger())
                greedy = q_now.masked_fill(~ok, float("-inf")).argmax(1)
                rand_a = torch.multinomial(ok.float(), 1).squeeze(1)   # explore among the moves that do not kill at once
            else:
                greedy, rand_a = q_now.argmax(1), torch.randint(0, 3, (a.envs,), device=dev)
        explore = torch.rand(a.envs, device=dev) < eps
        act = torch.where(explore, rand_a, greedy)
        nxt, rew, term, trunc, info = env.step(act)
        done = (term | trunc).float()
        tr = nstep.push(obs, act, rew, info["next_obs"], term, term | trunc, info["next_danger"])  # pre-reset next state
        if tr is not None:
            buf.add(*tr)
        obs = nxt
        steps += a.envs

        st["eps_n"] += done.sum()
        st["score_sum"] += (info["score"] * done).sum()
        st["score_max"] = torch.maximum(st["score_max"], (info["score"] * done).max())
        st["wins"] += info["won"].float().sum()
        st["causes"] += (F.one_hot(info["cause"], 5) * done.unsqueeze(1)).sum(0)

        if buf.size >= a.learn_start:
            for _ in range(a.updates_per_iter):
                beta = a.per_beta0 + (1.0 - a.per_beta0) * min(1.0, steps / a.total_steps)
                batch, idx, w = buf.sample(a.batch, beta)
                loss, qm, td_abs = td_loss(net, target, batch, w, not a.no_double, a.mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                buf.update_priorities(idx, td_abs)
                updates += 1
                st["loss_sum"] += loss.detach()
                st["q_sum"] += qm.detach()
                st["updates"] += 1
                if updates % a.target_every == 0:
                    target.load_state_dict(net.state_dict())

        if steps // a.epoch_steps != (steps - a.envs) // a.epoch_steps:  # crossed an epoch boundary
            epoch = steps // a.epoch_steps
            sync(dev)
            dt = time.time() - t_epoch
            n_ep = max(1.0, st["eps_n"].item())
            m = {"epoch": epoch, "env_steps": steps, "updates": updates, "eps": round(eps, 4),
                 "episodes": int(st["eps_n"].item()), "train_mean": st["score_sum"].item() / n_ep,
                 "train_max": int(st["score_max"].item()), "train_wins": int(st["wins"].item()),
                 "train_causes": {CAUSES[k]: int(st["causes"][k].item()) for k in range(1, 5)},
                 "loss": st["loss_sum"].item() / max(1, st["updates"]), "q_mean": st["q_sum"].item() / max(1, st["updates"]),
                 "steps_per_s": round(a.epoch_steps / dt), "seconds": round(dt, 1),
                 "vram_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20) if dev.type == "cuda" else None}
            print(f"ep {epoch:4d} | steps {steps / 1e6:6.2f}M | ε {eps:.3f} | train mean {m['train_mean']:6.2f} max {m['train_max']:3d} "
                  f"wins {m['train_wins']:3d} | loss {m['loss']:.4f} Q {m['q_mean']:6.2f} | {m['steps_per_s']:,} steps/s", flush=True)

            if epoch % a.checkpoint_every == 0:
                test = greedy_eval(net, TEST_SEEDS, a.size, dev, a.obs, a.mask)
                val = greedy_eval(net, VAL_SEEDS, a.size, dev, a.obs, a.mask)
                prev = checkpoints[-1]["test"]["mean"] if checkpoints else None
                cp = {"epoch": epoch, "env_steps": steps, "eps": round(eps, 4), "train_mean": m["train_mean"],
                      "train_max": m["train_max"], "train_wins": m["train_wins"],
                      "test": {k: v for k, v in test.items() if k != "scores"}, "val_mean": val["mean"],
                      "delta_test_mean": None if prev is None else test["mean"] - prev,
                      "best_fill_pct": round(100 * (test["max"] + 3) / a.size ** 2, 1)}
                checkpoints.append(cp)
                torch.save({"state_dict": net.state_dict(), "config": {"obs": a.obs, "size": a.size, "dueling": a.dueling, "mask": a.mask, "per": a.per, "double": not a.no_double},
                            "epoch": epoch, "test": cp["test"]}, RUNS / f"{name}.pt")
                if val["mean"] > best_val:  # select on validation seeds, report on test seeds
                    best_val = val["mean"]
                    torch.save({"state_dict": net.state_dict(), "config": {"obs": a.obs, "size": a.size, "dueling": a.dueling, "mask": a.mask, "per": a.per, "double": not a.no_double},
                                "epoch": epoch, "test": cp["test"], "val_mean": val["mean"]}, RUNS / f"{name}_best.pt")
                c = cp["test"]["causes"]
                d = "" if prev is None else f"{cp['delta_test_mean']:+.2f}"
                print(f"==== CHECKPOINT epoch {epoch}: TEST mean {cp['test']['mean']:.2f} median {cp['test']['median']:.0f} "
                      f"p90 {cp['test']['p90']:.0f} max {cp['test']['max']} wins {cp['test']['wins']}/500 Δ {d} | "
                      f"deaths {c} | val {val['mean']:.2f} | best fill {cp['best_fill_pct']}%", flush=True)
                m["checkpoint"] = cp
            log.write(json.dumps(m) + "\n")
            log.flush()
            metrics.append(m)
            st, t_epoch = stats0(), time.time()
            save_state(name, snapshot(it + 1))

    log.close()
    final = torch.load(RUNS / f"{name}_best.pt", map_location=dev) if checkpoints else None
    if final:
        net.load_state_dict(final["state_dict"])
    replays = record_replays(net, a.size, TEST_SEEDS[:5], dev, a.obs, a.mask)
    summary = {"tryout": "02_dqn", "name": name, "config": cfg, "epochs": metrics, "checkpoints": checkpoints,
               "best": {"epoch": final["epoch"], "val_mean": final["val_mean"], "test": final["test"]} if final else None,
               "replays": replays, "benchFile": "results/bench/bench.json",
               "baselines12": {"random": 0.168, "greedy_heuristic": 20.416, "tabular_q": 20.08}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    state_path(name).unlink(missing_ok=True)   # finished: the resume state is no longer needed
    notes_dir = ROOT / "notes" / "tryouts" / "02_dqn"
    if a.obs == "grid" and notes_dir.exists() and a.name is None:  # named runs (e.g. device comparisons) don't overwrite the notes data
        (notes_dir / "results.json").write_text(json.dumps(summary))
        print("wrote notes/tryouts/02_dqn/results.json (rebuild notes with: .\\run.ps1 notes)")
    print(f"done: {steps:,} env steps, {updates:,} updates. Models: models/_runs/{name}.pt, models/_runs/{name}_best.pt")
    if a.publish and checkpoints:
        from snake_rl.export import export_checkpoint
        export_checkpoint(RUNS / f"{name}_best.pt", a.publish, a.publish_name or a.publish, a.tryout, a.notes,
                          a.description, run=name, only_if_better=True)


if __name__ == "__main__":
    main()

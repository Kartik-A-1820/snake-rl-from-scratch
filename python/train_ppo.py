# Tryout 04: PPO (Schulman et al. 2017) on the batched torch vec env.
#   CNN actor-critic with a shared trunk, GAE(lambda), clipped surrogate objective, clipped value loss,
#   entropy bonus, per-minibatch advantage normalisation, orthogonal init, linear LR decay.
#   --mask: invalid-action masking (Huang & Ontanon 2020): logits of moves that kill at once are set to -inf before the
#   softmax, both when sampling and when computing log-probs / entropy for the update, so the gradient never pushes on them.
#
#   python python/train_ppo.py --total-steps 20000000 --publish 04_ppo --publish-name "PPO (CNN, masked)" --mask --tryout 4
#
# On-policy: every rollout (envs x horizon transitions) is used for K epochs of minibatch updates and then thrown away.
# That is why it wants many parallel envs, and why it suits an env that steps hundreds of thousands of games per second
# on the GPU: the data is nearly free, the update is the cost.
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from snake_rl.core import SnakeGame
from snake_rl.device import describe, pick_device, sync
from snake_rl.features import grid_obs, trap_danger, win_obs
from snake_rl.nets import ActorCritic, n_params
from snake_rl.resume import load_state, save_state, state_path, stop_requested
from snake_rl.vec_env import CAUSES, TorchSnakeVecEnv, evaluate

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models" / "_runs"
TEST_SEEDS = list(range(100000, 100500))   # same held-out seeds as tryouts 01-03
VAL_SEEDS = list(range(200000, 200200))    # pick the "best" checkpoint on these, report on TEST_SEEDS


def split_eval(res, a_end):
    """Statistics of one evaluate() result restricted to the games [0, a_end) and [a_end, n): lets the 500 test games and the 200
    validation games share ONE batched evaluation loop (the loop runs until the slowest game ends, so this roughly halves the cost)."""
    out = []
    for sl in (slice(0, a_end), slice(a_end, None)):
        sc = torch.tensor(res["scores"][sl], dtype=torch.float32)
        ci = torch.tensor(res["cause_ids"][sl])
        q = torch.quantile(sc, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        mid = sc.sort().values[int(len(sc) * 0.25): int(round(len(sc) * 0.75))]
        out.append({"n": len(sc), "mean": float(sc.mean()), "median": float(q[2]), "p10": float(q[0]), "p90": float(q[4]), "max": int(sc.max()),
                    "iqm": float(mid.mean()), "wins": int((ci == 4).sum()),
                    "causes": {CAUSES[k]: int((ci == k).sum()) for k in range(1, 5) if (ci == k).any()}, "scores": sc.tolist()})
    return out


def plateau_reached(checkpoints, steps, size, window, fill_gain, win_gain):
    """Stop rule for long runs. Looks at the checkpoint evaluations: those inside the last `window` steps against those before it.
    Plateau = the best mean fill inside the window improved by at most `fill_gain` percentage points over the best before it AND the best
    win rate (of 500 held-out games) improved by at most `win_gain` percentage points. With no wins at all this reduces to "fill stalled".
    Returns a message when the rule fires, else None."""
    cells = size * size - 3
    rows = [(c["env_steps"], 100 * c["test"]["mean"] / cells, 100 * c["test"]["wins"] / c["test"].get("n", 500)) for c in checkpoints]
    before = [(f, w) for s_, f, w in rows if s_ <= steps - window]
    inside = [(f, w) for s_, f, w in rows if s_ > steps - window]
    if not before or not inside:
        return None
    f_b, w_b = max(f for f, _ in before), max(w for _, w in before)
    f_i, w_i = max(f for f, _ in inside), max(w for _, w in inside)
    if f_i <= f_b + fill_gain and w_i <= w_b + win_gain:
        return (f"plateau over the last {window / 1e6:.0f} M steps: best fill {f_i:.1f}% (before: {f_b:.1f}%, gain limit {fill_gain} pt), "
                f"best win rate {w_i:.1f}% (before: {w_b:.1f}%, gain limit {win_gain} pt)")
    return None


def safe_moves(danger):
    ok = ~danger
    return ok | ~ok.any(1, keepdim=True)   # if every move kills, leave them all open


def masked_logits(logits, ok):
    return logits.masked_fill(~ok, float("-inf")) if ok is not None else logits


def greedy_eval(net, seeds, size, device, mask, obs="grid", trap=False):
    net.eval()
    if mask:
        res = evaluate(lambda x, d: masked_logits(net(x), safe_moves(d)).argmax(1), seeds, size=size, device=device,
                       obs=obs, masked=True, trap=trap)
    else:
        res = evaluate(lambda x: net(x).argmax(1), seeds, size=size, device=device, obs=obs)
    net.train()
    return res


@torch.no_grad()
def record_replays(net, size, seeds, obs, mask, limit=3, trap=False):
    """Re-play held-out seeds with the scalar engine on the CPU and return {seed, width, height, actions, score, cause} for each
    (the game's "Replay a win" button plays these). Used for the seeds the greedy policy won."""
    net = net.cpu().eval()
    out = []
    for seed in seeds[:limit]:
        g, acts = SnakeGame(size, size, seed=seed), []
        while not g.done:
            x = {"grid": lambda: grid_obs(g), "win5": lambda: win_obs(g), "win6": lambda: win_obs(g, True)}[obs]()
            q = net(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))[0].tolist()
            bad = (trap_danger(g) if trap else basic_danger(g)) if mask else [0, 0, 0]
            if all(bad):
                bad = [0, 0, 0]
            a = max((i for i in range(3) if not bad[i]), key=lambda i: (q[i], -i))
            acts.append(a)
            g.step(a)
        out.append({"seed": seed, "width": size, "height": size, "actions": "".join(map(str, acts)), "score": g.score, "cause": g.cause})
    return out


def basic_danger(g):
    from snake_rl.features import basic_features
    return basic_features(g)[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--obs", choices=["grid", "win5", "win6"], default="grid",
                    help="grid: 4x14x14 board image (tryout 04). win5/win6: fixed 14x14 canvas with tail order, head, food, wall, "
                         "length (+ reachable area), so one network plays every board up to 12x12 (tryout 05)")
    ap.add_argument("--win-reward", type=float, default=0.0, help="extra reward for filling the whole board")
    ap.add_argument("--init-from", help="checkpoint (.pt) to start from: curriculum fine-tuning on a larger board")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="continue from models/_runs/<name>_state.pt (see snake_rl/resume.py)")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=64, help="rollout steps per env before each update")
    ap.add_argument("--total-steps", type=int, default=20_000_000)
    ap.add_argument("--epoch-steps", type=int, default=500_000, help="env steps per log line")
    ap.add_argument("--checkpoint-every", type=int, default=5, help="log epochs between full evaluations")
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--no-lr-decay", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--no-value-clip", action="store_true")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.0, help="stop the update epochs early above this KL (0 = never)")
    ap.add_argument("--mask", action="store_true", help="invalid-action masking of moves that kill on the next tick")
    ap.add_argument("--trap-mask", action="store_true", help="stronger mask (implies --mask): also moves into a region too small for the snake; planning built into the agent, see tryout 05")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--publish", metavar="ALGO_ID")
    ap.add_argument("--publish-live", action="store_true",
                    help="long runs: at every new best checkpoint, export it in a background process and publish it (marked in training) "
                         "to the registry, at most once per 20 minutes")
    ap.add_argument("--plateau-stop", action="store_true",
                    help="stop when the best mean fill has not improved by more than --plateau-gain points over the last --plateau-window "
                         "steps AND no checkpoint in that window had a win")
    ap.add_argument("--plateau-window", type=int, default=15_000_000)
    ap.add_argument("--plateau-gain", type=float, default=1.0, help="fill percentage points")
    ap.add_argument("--plateau-win-gain", type=float, default=2.0, help="win-rate percentage points")
    ap.add_argument("--stop-at", metavar="HH:MM", help="end the run (a normal finish: summary and final publish) at the next HH:MM local time")
    ap.add_argument("--publish-name")
    ap.add_argument("--tryout", type=int)
    ap.add_argument("--notes")
    ap.add_argument("--description", default="")
    a = ap.parse_args()
    a.mask = a.mask or a.trap_mask   # the trap mask is an extension of the instant-death mask

    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    name = a.name or f"ppo_cnn_{a.size}x{a.size}{'_mask' if a.mask else ''}"
    run_dir = ROOT / "results" / "phase4_ppo" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    N, T = a.envs, a.horizon
    batch = N * T
    mb = batch // a.minibatches
    env = TorchSnakeVecEnv(N, a.size, seeds=[10_000_000 * (a.seed + 1) + i for i in range(N)], device=dev, obs=a.obs,
                           gamma=a.gamma, win_reward=a.win_reward)
    in_ch = {"grid": 4, "win5": 5, "win6": 6}[a.obs]
    net_size = a.size if a.obs == "grid" else 12     # the canvas is 14x14 whatever the board size
    net = ActorCritic(net_size, in_ch=in_ch).to(dev)
    if a.init_from:
        net.load_state_dict(torch.load(a.init_from, map_location=dev)["state_dict"])
        print(f"[init] weights from {a.init_from}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    obs = env.obs()
    cfg = dict(vars(a), device=describe(dev), params=n_params(net), batch=batch, minibatch=mb)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"[model] {name}: {n_params(net):,} params · {N} envs x {T} steps = {batch:,} per update, {a.update_epochs} epochs x "
          f"{a.minibatches} minibatches of {mb:,}", flush=True)

    shape = obs.shape[1:]
    obs_b = torch.zeros((T, N, *shape), device=dev)
    act_b = torch.zeros((T, N), dtype=torch.long, device=dev)
    logp_b = torch.zeros((T, N), device=dev)
    val_b = torch.zeros((T, N), device=dev)
    rew_b = torch.zeros((T, N), device=dev)
    done_b = torch.zeros((T, N), device=dev)
    ok_b = torch.ones((T, N, 3), dtype=torch.bool, device=dev)

    def stats0():
        z = lambda: torch.zeros((), device=dev)
        return {"eps_n": z(), "score_sum": z(), "score_max": z(), "wins": z(), "causes": torch.zeros(5, device=dev),
                "pl": z(), "vl": z(), "ent": z(), "kl": z(), "cf": z(), "n": 0}

    st, metrics, checkpoints = stats0(), [], []
    first_win = None
    live = {"proc": None, "t": 0.0}
    stopped = None
    steps = updates = 0
    best_val = -1.0
    t_epoch = time.time()
    n_iters = a.total_steps // batch
    it0 = 0
    saved = load_state(name, dev) if a.resume else None
    if saved:
        net.load_state_dict(saved["net"])
        opt.load_state_dict(saved["opt"])
        steps, updates, it0, best_val, first_win = saved["steps"], saved["updates"], saved["it"], saved["best_val"], saved["first_win"]
        metrics, checkpoints = saved["metrics"], saved["checkpoints"]
        print(f"[resume] {name}: iteration {it0}/{n_iters}, {steps / 1e6:.2f} M steps, {len(checkpoints)} checkpoints so far", flush=True)
    elif a.resume:
        print(f"[resume] no saved state for {name}: starting from the beginning", flush=True)
    log = open(run_dir / "metrics.jsonl", "a" if saved else "w")
    next_log = (steps // a.epoch_steps + 1) * a.epoch_steps

    def snapshot(next_it):
        return {"net": net.state_dict(), "opt": opt.state_dict(), "steps": steps, "updates": updates, "it": next_it, "best_val": best_val,
                "first_win": first_win, "metrics": metrics, "checkpoints": checkpoints}

    def save(path, extra):
        torch.save({"state_dict": net.state_dict(),
                    "config": {"obs": a.obs, "size": a.size, "dueling": False, "mask": a.mask, "trap_mask": a.trap_mask, "algo": "ppo", "in_ch": in_ch,
                               "net_size": net_size}, **extra}, path)

    deadline = None
    if a.stop_at:
        import datetime as _dt
        hh, mm = map(int, a.stop_at.split(":"))
        t = _dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        deadline = (t if t > _dt.datetime.now() else t + _dt.timedelta(days=1)).timestamp()
        print(f"[deadline] the run ends at {_dt.datetime.fromtimestamp(deadline):%Y-%m-%d %H:%M}", flush=True)
    for it in range(it0, n_iters):
        if deadline and time.time() >= deadline:
            stopped = f"deadline {a.stop_at} reached at {steps / 1e6:.1f} M steps"
            print(f"[deadline] {stopped}", flush=True)
            break
        if stop_requested(name):
            save_state(name, snapshot(it))
            log.close()
            print(f"[stop] stop file found: state saved to {state_path(name)} at iteration {it} ({steps / 1e6:.2f} M steps). "
                  f"Resume with --resume.", flush=True)
            return
        if not a.no_lr_decay:
            for g in opt.param_groups:
                g["lr"] = a.lr * (1.0 - it / n_iters)

        # ---- rollout: no gradients, T steps in all N envs ------------------------------------------------------
        with torch.no_grad():
            for t in range(T):
                logits, v = net.forward_ac(obs)
                ok = safe_moves(env.danger_trap() if a.trap_mask else env.danger()) if a.mask else None
                logits = masked_logits(logits, ok)
                logp_all = F.log_softmax(logits, dim=-1)
                act = torch.multinomial(logp_all.exp(), 1).squeeze(1)
                obs_b[t], act_b[t], val_b[t] = obs, act, v
                logp_b[t] = logp_all.gather(1, act.unsqueeze(1)).squeeze(1)
                if a.mask:
                    ok_b[t] = ok
                nxt, rew, term, trunc, info = env.step(act, lazy_next=True)
                # Time-limit truncation is not a real terminal: fold gamma*V(s') into the reward and cut the GAE trace.
                tr = trunc & ~term
                if info["trunc_any"]:
                    rew = rew + a.gamma * net.value(info["next_obs"]) * tr.float()
                rew_b[t], done_b[t] = rew, (term | trunc).float()
                done = (term | trunc).float()
                st["eps_n"] += done.sum()
                st["score_sum"] += (info["score"] * done).sum()
                st["score_max"] = torch.maximum(st["score_max"], (info["score"] * done).max())
                st["wins"] += info["won"].float().sum()
                st["causes"] += (F.one_hot(info["cause"], 5) * done.unsqueeze(1)).sum(0)
                obs = nxt
            last_v = net.value(obs)
            # ---- GAE(lambda): A_t = sum_l (gamma*lambda)^l delta_{t+l},  delta_t = r_t + gamma V(s_{t+1}) - V(s_t) ----
            adv = torch.zeros_like(rew_b)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else val_b[t + 1]
                nonterm = 1.0 - done_b[t]
                delta = rew_b[t] + a.gamma * nv * nonterm - val_b[t]
                gae = delta + a.gamma * a.gae_lambda * nonterm * gae
                adv[t] = gae
            ret = adv + val_b
        steps += batch

        # ---- update: K epochs over shuffled minibatches ------------------------------------------------------------
        b_obs, b_act = obs_b.reshape(batch, *shape), act_b.reshape(batch)
        b_logp, b_val, b_adv, b_ret = logp_b.reshape(batch), val_b.reshape(batch), adv.reshape(batch), ret.reshape(batch)
        b_ok = ok_b.reshape(batch, 3) if a.mask else None
        stop = False
        for _ in range(a.update_epochs):
            perm = torch.randperm(batch, device=dev)
            for i in range(a.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                logits, v = net.forward_ac(b_obs[idx])
                logits = masked_logits(logits, b_ok[idx] if a.mask else None)
                logp_all = F.log_softmax(logits, dim=-1)
                logp = logp_all.gather(1, b_act[idx].unsqueeze(1)).squeeze(1)
                p = logp_all.exp()
                ent = -(p * logp_all.masked_fill(p == 0, 0.0)).sum(-1).mean()
                ratio = (logp - b_logp[idx]).exp()
                madv = b_adv[idx]
                madv = (madv - madv.mean()) / (madv.std() + 1e-8)           # advantage normalisation per minibatch
                pg = torch.max(-madv * ratio, -madv * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()   # clipped surrogate
                if a.no_value_clip:
                    vl = 0.5 * (v - b_ret[idx]).pow(2).mean()
                else:                                                         # clipped value loss
                    vc = b_val[idx] + (v - b_val[idx]).clamp(-a.clip, a.clip)
                    vl = 0.5 * torch.max((v - b_ret[idx]).pow(2), (vc - b_ret[idx]).pow(2)).mean()
                loss = pg + a.vf_coef * vl - a.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm)
                opt.step()
                updates += 1
                with torch.no_grad():
                    lr_ = logp - b_logp[idx]
                    kl = ((lr_.exp() - 1) - lr_).mean()                       # k3 estimator of KL(old || new)
                    st["pl"] += pg.detach(); st["vl"] += vl.detach(); st["ent"] += ent.detach(); st["kl"] += kl
                    st["cf"] += ((ratio - 1).abs() > a.clip).float().mean(); st["n"] += 1
            if a.target_kl > 0 and float(kl) > a.target_kl:
                stop = True
            if stop:
                break

        # ---- logging / evaluation ---------------------------------------------------------------------------------
        if steps >= next_log:
            next_log += a.epoch_steps
            epoch = steps // a.epoch_steps
            sync(dev)
            dt = time.time() - t_epoch
            n_ep = max(1.0, st["eps_n"].item())
            n = max(1, st["n"])
            m = {"epoch": epoch, "env_steps": steps, "updates": updates, "episodes": int(st["eps_n"].item()),
                 "train_mean": st["score_sum"].item() / n_ep, "train_max": int(st["score_max"].item()),
                 "train_wins": int(st["wins"].item()),
                 "train_causes": {CAUSES[k]: int(st["causes"][k].item()) for k in range(1, 5)},
                 "loss": st["vl"].item() / n, "pg_loss": st["pl"].item() / n, "entropy": st["ent"].item() / n,
                 "approx_kl": st["kl"].item() / n, "clipfrac": st["cf"].item() / n, "q_mean": float(b_val.mean()),
                 "steps_per_s": round(a.epoch_steps / dt), "seconds": round(dt, 1),
                 "vram_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20) if dev.type == "cuda" else None}
            print(f"ep {epoch:4d} | steps {steps / 1e6:6.2f}M | train mean {m['train_mean']:6.2f} max {m['train_max']:3d} "
                  f"| ent {m['entropy']:.3f} kl {m['approx_kl']:.4f} clip {m['clipfrac']:.3f} V {m['loss']:.4f} | "
                  f"{m['steps_per_s']:,} steps/s", flush=True)
            if epoch % a.checkpoint_every == 0:
                both = greedy_eval(net, TEST_SEEDS + VAL_SEEDS, a.size, dev, a.mask, a.obs, a.trap_mask)
                test, val = split_eval(both, len(TEST_SEEDS))
                if test["wins"] and first_win is None:
                    won_seeds = [sd for sd, sc in zip(TEST_SEEDS, test["scores"]) if sc == a.size * a.size - 3]
                    first_win = {"epoch": epoch, "env_steps": steps, "replays": record_replays(net, a.size, won_seeds, a.obs, a.mask, trap=a.trap_mask)}
                    net.to(dev)
                    print(f"**** FIRST WIN on the test seeds at epoch {epoch} ({steps / 1e6:.2f}M steps): {test['wins']}/500", flush=True)
                prev = checkpoints[-1]["test"]["mean"] if checkpoints else None
                cp = {"epoch": epoch, "env_steps": steps, "eps": 0.0, "train_mean": m["train_mean"],
                      "train_max": m["train_max"], "train_wins": m["train_wins"],
                      "test": {k: v for k, v in test.items() if k != "scores"}, "val_mean": val["mean"],
                      "delta_test_mean": None if prev is None else test["mean"] - prev,
                      "best_fill_pct": round(100 * (test["max"] + 3) / a.size ** 2, 1)}
                checkpoints.append(cp)
                save(RUNS / f"{name}.pt", {"epoch": epoch, "test": cp["test"]})
                score_key = val["wins"] * 1000.0 + val["mean"]       # tryout 05: win rate dominates, mean fills the ties
                if score_key > best_val:
                    best_val = score_key
                    save(RUNS / f"{name}_best.pt", {"epoch": epoch, "test": cp["test"], "val_mean": val["mean"], "val_wins": val["wins"]})
                    if test["wins"]:   # keep winning games of the best checkpoint for the game's "Replay a win" button
                        won = [sd for sd, sc in zip(TEST_SEEDS, test["scores"]) if sc == a.size * a.size - 3]
                        (run_dir / "best_replays.json").write_text(json.dumps(record_replays(net, a.size, won, a.obs, a.mask, limit=1, trap=a.trap_mask)))
                        net.to(dev)
                    if a.publish and a.publish_live and (live["proc"] is None or live["proc"].poll() is not None) and time.time() - live["t"] > 1200:
                        import shutil
                        shutil.copyfile(RUNS / f"{name}_best.pt", RUNS / f"{name}_live.pt")
                        cmd = [sys.executable, str(ROOT / "python" / "export_model.py"), "--ckpt", str(RUNS / f"{name}_live.pt"), "--algo", a.publish,
                               "--name", a.publish_name or a.publish, "--tryout", str(a.tryout or 5), "--run", name, "--provisional",
                               "--description", (a.description + " ").lstrip() + f"Extended run in progress: best checkpoint so far (epoch {epoch}, {steps / 1e6:.0f} M steps)."]
                        if a.notes:
                            cmd += ["--notes", a.notes]
                        live["proc"], live["t"] = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), time.time()
                c = cp["test"]["causes"]
                print(f"==== CHECKPOINT epoch {epoch}: TEST mean {cp['test']['mean']:.2f} median {cp['test']['median']:.0f} "
                      f"p90 {cp['test']['p90']:.0f} max {cp['test']['max']} wins {cp['test']['wins']}/500 | deaths {c} | "
                      f"val {val['mean']:.2f} | best fill {cp['best_fill_pct']}%", flush=True)
                m["checkpoint"] = cp
            log.write(json.dumps(m) + "\n")
            log.flush()
            metrics.append(m)
            st, t_epoch = stats0(), time.time()
            save_state(name, snapshot(it + 1))
            if a.plateau_stop and "checkpoint" in m:
                why = plateau_reached(checkpoints, steps, a.size, a.plateau_window, a.plateau_gain, a.plateau_win_gain)
                if why:
                    stopped = why
                    print(f"[plateau] {why}. Stopping at {steps / 1e6:.1f} M steps.", flush=True)
                    break

    log.close()
    summary = {"tryout": "04_ppo", "name": name, "config": cfg, "epochs": metrics, "checkpoints": checkpoints,
               "best": None, "first_win": first_win, "stopped": stopped}
    if checkpoints:
        final = torch.load(RUNS / f"{name}_best.pt", map_location=dev)
        summary["best"] = {"epoch": final["epoch"], "val_mean": final["val_mean"], "test": final["test"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    state_path(name).unlink(missing_ok=True)   # finished: the resume state is no longer needed
    print(f"done: {steps:,} env steps, {updates:,} updates. Models: models/_runs/{name}.pt, models/_runs/{name}_best.pt", flush=True)
    if a.publish and checkpoints:
        from snake_rl.export import export_checkpoint
        export_checkpoint(RUNS / f"{name}_best.pt", a.publish, a.publish_name or a.publish, a.tryout, a.notes,
                          a.description, run=name, only_if_better=True)


if __name__ == "__main__":
    main()

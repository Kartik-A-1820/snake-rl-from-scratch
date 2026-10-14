# Static figures/tables for the DOCX version of tryout 02, from the same results.json as the HTML.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def make(R, out):
    figs = {}
    if not R.get("epochs"):
        return figs
    ep = [e["epoch"] for e in R["epochs"]]
    cps = R["checkpoints"]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(ep, [e["train_mean"] for e in R["epochs"]], "--", color="#c27c0e", label="train mean (ε-greedy)")
    ax.plot([c["epoch"] for c in cps], [c["test"]["mean"] for c in cps], "o-", color="#306230", label="TEST mean (greedy, 500 seeds)")
    ax.axhline(R["baselines12"]["greedy_heuristic"], color="#b3261e", ls=":", label="heuristic / tabular Q ≈ 20")
    ax.set(xlabel="epoch (50k env steps)", ylabel="score")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=160)
    plt.close(fig)
    figs["curves"] = {"png": out / "curves.png", "caption": "DQN learning curve on 12×12."}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(ep, [e["loss"] for e in R["epochs"]], color="#3a6ea5")
    axes[0].set(title="Huber TD loss", xlabel="epoch")
    axes[1].plot(ep, [e["q_mean"] for e in R["epochs"]], color="#9b3d8f")
    axes[1].set(title="mean Q(s,a) on sampled batches", xlabel="epoch")
    for a in axes:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "loss_q.png", dpi=160)
    plt.close(fig)
    figs["loss_q"] = {"png": out / "loss_q.png", "caption": "Training health: loss and predicted Q."}

    rows = ["| epoch | env steps | ε | train mean (max, wins) | TEST mean | median | p90 | max | wins | Δ test | val mean |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cps:
        d = "–" if c["delta_test_mean"] is None else f"{c['delta_test_mean']:+.2f}"
        t = c["test"]
        rows.append(f"| {c['epoch']} | {c['env_steps'] / 1e6:.2f}M | {c['eps']} | {c['train_mean']:.2f} ({c['train_max']}, {c['train_wins']}) | "
                    f"**{t['mean']:.2f}** | {t['median']:.0f} | {t['p90']:.0f} | {t['max']} | {t['wins']} | {d} | {c['val_mean']:.2f} |")
    figs["checkpoint_table"] = {"md": "\n".join(rows) + "\n"}

    c = R["config"]
    figs["summary"] = {"md": "\n".join([
        "| item | value |", "|---|---|",
        f"| model | {R['name']}: {c['params']:,} params, {c['model_bytes_fp32'] / 1e6:.2f} MB fp32 |",
        f"| device | {c['device']} |",
        f"| replay | {c['buffer']:,} transitions, {c['replay_bytes'] / 1e6:.0f} MB |",
        f"| best by validation | epoch {R['best']['epoch']}: test mean {R['best']['test']['mean']:.2f}, max {R['best']['test']['max']}, wins {R['best']['test']['wins']} |" if R.get("best") else "| best | – |",
    ]) + "\n"}

    m, sig, n = 3, 1.0, 200000
    rng = np.random.default_rng(0)
    qa, qb = rng.normal(0, sig, (n, m)), rng.normal(0, sig, (n, m))
    single, double = qa.max(1), qb[np.arange(n), qa.argmax(1)]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(single, 60, alpha=.6, color="#b3261e", label=f"max: mean {single.mean():.3f}")
    ax.hist(double, 60, alpha=.6, color="#306230", label=f"double: mean {double.mean():.3f}")
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="estimate (true value 0)", title="3 actions, σ = 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "overestimation.png", dpi=160)
    plt.close(fig)
    figs["overestimation"] = {"png": out / "overestimation.png", "caption": "Maximisation bias vs the Double estimator (Monte Carlo)."}

    B = R.get("bench")
    if B:
        lines = ["| model | device | threads | latency (µs, batch 1) |", "|---|---|---:|---:|"]
        lines += [f"| {r['model']} | {r['device']} | {r['threads'] or ''} | {r['latency_us']} |" for r in B["inference"]]
        lines += ["", "| model | device | batch | ms/step | samples/s |", "|---|---|---:|---:|---:|"]
        lines += [f"| {r['model']} | {r['device']} | {r['batch']} | {r['ms_per_step']} | {r['samples_per_s']:,} |" for r in B["train_step"]]
        lines += ["", "| env | device | n_envs | steps/s |", "|---|---|---:|---:|"]
        lines += [f"| {r['env']} | {r['device']} | {r['n_envs']} | {r['steps_per_s']:,} |" for r in B["env"]]
        figs["bench"] = {"md": "\n".join(lines) + "\n"}
    return figs

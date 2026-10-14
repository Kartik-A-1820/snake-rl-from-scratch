# Static figures/tables for the DOCX version of tryout 03, from the same results.json as the HTML.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#306230", "#c27c0e", "#3a6ea5", "#9b3d8f", "#b3261e", "#5b6b2f"]


def make(R, out):
    figs = {}
    arms = R.get("arms") or []

    if R.get("config"):
        c, b = R["config"], R.get("best")
        last = R["checkpoints"][-1] if R.get("checkpoints") else None
        rows = ["| item | value |", "|---|---|",
                f"| model | {R['name']}: {c['params']:,} params |", f"| device | {c['device']} |",
                f"| components | Double DQN{' + dueling' if c.get('dueling') else ''}{' + prioritized replay' if c.get('per') else ''}"
                f"{' + action masking' if c.get('mask') else ''}, {c.get('n_step', 1)}-step returns |",
                f"| training | {R['epochs'][-1]['env_steps'] / 1e6:.1f} M env steps, {sum(e['seconds'] for e in R['epochs']) / 60:.0f} min |"]
        if last:
            rows.append(f"| last checkpoint (test) | mean {last['test']['mean']:.2f}, median {last['test']['median']:.0f}, max {last['test']['max']} |")
        if b:
            rows.append(f"| best by validation | epoch {b['epoch']}: test mean {b['test']['mean']:.2f}, max {b['test']['max']}, deaths {b['test']['causes']} |")
        figs["summary"] = {"md": "\n".join(rows) + "\n"}

    if arms:
        rows = ["| arm | seeds | test mean per seed | mean ± sd | max | deaths wall / self / timeout (mean per 500) |", "|---|---:|---|---:|---:|---|"]
        for a in arms:
            m = [r["best"]["test"]["mean"] for r in a["runs"]]
            mx = max(r["best"]["test"]["max"] for r in a["runs"])
            cs = [round(np.mean([r["best"]["test"]["causes"].get(k, 0) for r in a["runs"]])) for k in ("wall", "self", "timeout")]
            rows.append(f"| {a['label']} | {len(m)} | {', '.join(f'{x:.1f}' for x in m)} | {np.mean(m):.1f} ± {np.std(m, ddof=1) if len(m) > 1 else 0:.1f} | {mx} | {' / '.join(map(str, cs))} |")
        figs["ablation_table"] = {"md": "\n".join(rows) + "\n"}

        fig, ax = plt.subplots(figsize=(8, 3.8))
        for i, a in enumerate(arms):
            n = min(len(r["curve"]) for r in a["runs"])
            xs = [r[0] * 0.05 for r in a["runs"][0]["curve"][:n]]
            ys = np.mean([[c[1] for c in r["curve"][:n]] for r in a["runs"]], axis=0)
            ax.plot(xs, ys, color=COLORS[i % len(COLORS)], label=f"{a['label']} (n={len(a['runs'])})")
        ax.axhline(20, color="#b3261e", ls=":", label="tabular ≈ 20")
        ax.set(xlabel="env steps (millions)", ylabel="greedy test mean (500 games)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "ablation_curves.png", dpi=160)
        plt.close(fig)
        figs["ablation_curves"] = {"png": out / "ablation_curves.png", "caption": "Ablation: mean test score over seeds at each checkpoint."}

    if R.get("epochs"):
        ep = [e["epoch"] * 0.05 for e in R["epochs"]]
        cps = R["checkpoints"]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(ep, [e["train_mean"] for e in R["epochs"]], "--", color="#c27c0e", label="train mean (ε-greedy)")
        ax.plot([c["epoch"] * 0.05 for c in cps], [c["test"]["mean"] for c in cps], "o-", color="#306230", label="TEST mean (greedy, 500 seeds)")
        ax.axhline(20, color="#b3261e", ls=":", label="tabular ≈ 20")
        ax.set(xlabel="env steps (millions)", ylabel="score")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "long_curves.png", dpi=160)
        plt.close(fig)
        figs["long_curves"] = {"png": out / "long_curves.png", "caption": "Long run of the full combination."}

    # static version of the PER demo: sampling probability and importance weight for 12 increasing TD errors
    delta = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for alpha, col in ((0.0, "#888"), (0.6, "#306230"), (1.0, "#b3261e")):
        p = (delta + 1e-6) ** alpha
        P = p / p.sum()
        axes[0].plot(range(12), P, "o-", color=col, label=f"α = {alpha}")
        w = (1 / (12 * P)) ** 0.4
        axes[1].plot(range(12), w / w.max(), "o-", color=col, label=f"α = {alpha}, β = 0.4")
    axes[0].set(title="sampling probability P(i)", xlabel="transition (TD error ↑)")
    axes[1].set(title="importance weight w_i", xlabel="transition (TD error ↑)")
    for a in axes:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "per_demo.png", dpi=160)
    plt.close(fig)
    figs["per_demo"] = {"png": out / "per_demo.png", "caption": "Prioritized replay: larger TD errors are sampled more often and weighted less."}
    return figs

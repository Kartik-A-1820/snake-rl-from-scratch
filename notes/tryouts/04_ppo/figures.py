# Static figures/tables for the DOCX version of tryout 04, from the same results.json as the HTML.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#306230", "#c27c0e", "#3a6ea5", "#9b3d8f", "#b3261e"]


def make(R, out):
    figs = {}
    arms = R.get("arms") or []

    if R.get("config"):
        c, b = R["config"], R.get("best")
        last = R["checkpoints"][-1] if R.get("checkpoints") else None
        rows = ["| item | value |", "|---|---|", f"| model | {R['name']}: {c['params']:,} params (shared CNN trunk) |", f"| device | {c['device']} |",
                f"| setup | {c['envs']} envs × {c['horizon']} steps = {c['batch']:,} per update, {c['update_epochs']} epochs × {c['minibatches']} minibatches, "
                f"lr {c['lr']}, γ {c['gamma']}, λ {c['gae_lambda']}, clip {c['clip']}, entropy {c['ent_coef']}{', action masking' if c.get('mask') else ''} |",
                f"| training | {R['epochs'][-1]['env_steps'] / 1e6:.1f} M env steps in {sum(e['seconds'] for e in R['epochs']) / 60:.0f} min |"]
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
        figs["ppo_table"] = {"md": "\n".join(rows) + "\n"}

        fig, ax = plt.subplots(figsize=(8, 3.6))
        for i, a in enumerate(arms):
            n = min(len(r["curve"]) for r in a["runs"])
            xs = [r[0] * 0.5 for r in a["runs"][0]["curve"][:n]]
            ax.plot(xs, np.mean([[c[1] for c in r["curve"][:n]] for r in a["runs"]], axis=0), color=COLORS[i % 5], label=f"{a['label']} (n={len(a['runs'])})")
        ax.axhline(20, color="#b3261e", ls=":", label="tabular ≈ 20")
        ax.set(xlabel="env steps (millions)", ylabel="greedy test mean (500 games)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "ppo_curves.png", dpi=160)
        plt.close(fig)
        figs["ppo_curves"] = {"png": out / "ppo_curves.png", "caption": "PPO learning curves (mean over seeds)."}

    par = R.get("parallel") or {}
    if par:
        rows = ["| family | jobs | steps/s per job | aggregate steps/s | speed-up | GPU util % | CPU util % | GPU memory in use (MB) |", "|---|---:|---|---:|---:|---:|---:|---:|"]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for i, (fam, P) in enumerate(par.items()):
            base = P["rows"][0]["aggregate_steps_per_s"]
            for r in P["rows"]:
                rows.append(f"| {fam.upper()} | {r['jobs']} | {' / '.join(f'{v:,}' for v in r['per_job_steps_per_s'])} | {r['aggregate_steps_per_s']:,} | "
                            f"{r['aggregate_steps_per_s'] / base:.2f}× | {r['gpu_util_pct']} | {r['cpu_util_pct']} | {r['gpu_mem_used_mb']} |")
            ax.plot([r["jobs"] for r in P["rows"]], [r["aggregate_steps_per_s"] / base for r in P["rows"]], "o-", color=COLORS[i * 2], label=fam.upper())
        ax.axhline(1, color="#888", ls=":")
        ax.set(xlabel="concurrent jobs", ylabel="aggregate throughput vs 1 job", ylim=(0, None))
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "parallel.png", dpi=160)
        plt.close(fig)
        figs["parallel"] = {"md": "\n".join(rows) + f"\n\n![Aggregate throughput against the number of concurrent jobs.]({(out / 'parallel.png').as_posix()}){{width=70%}}\n"}

    # GAE weights and the PPO clipped objective, as static plots
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for lam, col in ((0.0, "#888"), (0.5, "#3a6ea5"), (0.95, "#306230"), (1.0, "#b3261e")):
        axes[0].plot(range(30), [(0.99 * lam) ** l for l in range(30)], color=col, label=f"λ = {lam}")
    axes[0].set(title="GAE weight on δ_{t+l}", xlabel="l")
    r = np.linspace(0.4, 1.6, 200)
    eps = 0.2
    for A, col, lab in ((1, "#306230", "A > 0"), (-1, "#b3261e", "A < 0")):
        axes[1].plot(r, np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A), color=col, label=lab)
    axes[1].axvline(1 - eps, color="#c27c0e", ls=":")
    axes[1].axvline(1 + eps, color="#c27c0e", ls=":")
    axes[1].set(title="clipped objective (ε = 0.2)", xlabel="ratio r")
    for a in axes:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gae_clip.png", dpi=160)
    plt.close(fig)
    figs["gae_demo"] = {"png": out / "gae_clip.png", "caption": "Left: GAE weights for several λ. Right: PPO's clipped surrogate."}
    figs["clip_demo"] = {"md": ""}
    return figs

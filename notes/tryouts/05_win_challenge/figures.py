# Static figures/tables for the DOCX version of tryout 05, from the same results.json as the HTML.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#306230", "#c27c0e", "#3a6ea5", "#9b3d8f", "#b3261e"]


def make(R, out):
    figs = {}
    stages = R.get("stages") or []

    # discounting figure (static version of the slider)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    t = np.arange(0, 801)
    for g, col in ((0.99, "#888"), (0.995, "#c27c0e"), (0.999, "#3a6ea5")):
        ax.plot(t, g ** t, color=col, label=f"γ = {g} (horizon {1 / (1 - g):.0f})")
    ax.axvline(300, color="#b3261e", ls=":")
    ax.set(xlabel="steps T until the reward", ylabel="weight γ^T")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "horizon.png", dpi=160)
    plt.close(fig)
    figs["horizon_demo"] = {"png": out / "horizon.png", "caption": "A reward T steps away is weighted by γ^T; the dotted line marks a 6×6-sized win (T ≈ 300)."}

    if stages:
        rows = ["| board | run | start from | win rate | mean score (fill %) | max | wins / games | deaths wall / self / timeout | steps |",
                "|---|---|---|---:|---:|---:|---:|---|---:|"]
        for st in stages:
            cells = st["size"] ** 2 - 3
            for r in st["runs"]:
                t_ = r["best"]["test"]
                c = t_.get("causes", {})
                rows.append(f"| {st['size']}×{st['size']} | {r['name']} | {r['init_from'] or 'scratch'} | **{100 * t_['wins'] / t_.get('n', 500):.1f}%** | "
                            f"{t_['mean']:.1f} ({100 * t_['mean'] / cells:.0f}%) | {t_['max']} | {t_['wins']} / {t_.get('n', 500)} | "
                            f"{c.get('wall', 0)} / {c.get('self', 0)} / {c.get('timeout', 0)} | {r['steps'] / 1e6:.1f} M |")
        if R.get("hamilton"):
            rows.append("")
            rows.append("Reference, non-RL Hamiltonian cycle: " + " · ".join(f"{h['size']}×{h['size']} {100 * h['win_rate']:.0f}%" for h in R["hamilton"]))
        figs["curriculum_table"] = {"md": "\n".join(rows) + "\n"}
        figs["summary"] = figs["curriculum_table"]

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
        for i, st in enumerate(stages):
            for j, r in enumerate(st["runs"]):
                lab = f"{st['size']}×{st['size']}" + (" scratch" if r["scratch"] else "")
                xs = [c[0] * 0.5 for c in r["curve"]]
                axes[0].plot(xs, [100 * c[2] / c[3] for c in r["curve"]], color=COLORS[(i + j) % 5], label=lab)
                axes[1].plot(xs, [100 * c[1] / (st["size"] ** 2 - 3) for c in r["curve"]], color=COLORS[(i + j) % 5], label=lab)
        axes[0].set(title="test win rate (%)", xlabel="env steps of that stage (millions)")
        axes[1].set(title="mean fill (% of cells)", xlabel="env steps of that stage (millions)")
        for a in axes:
            a.grid(alpha=.3)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "win_curves.png", dpi=160)
        plt.close(fig)
        figs["win_curves"] = {"png": out / "win_curves.png", "caption": "Win rate and fill during training, per curriculum stage."}
    return figs

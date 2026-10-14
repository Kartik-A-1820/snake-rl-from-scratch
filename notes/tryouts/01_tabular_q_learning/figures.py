# Static versions of the interactive figures, for the DOCX note. Same results.json as the HTML.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = ["#306230", "#c27c0e", "#3a6ea5"]


def make(R, out):
    figs = {}
    ep = R["seedCurves"]["epochs"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, board in zip(axes, ["12x12", "6x6"]):
        for i, (k, ys) in enumerate(R["seedCurves"][board].items()):
            ax.plot(ep, ys, "o-", color=C[i], label=f"eval {k}")
        run = R["runs"][f"{board}_seed1"]["checkpoints"]
        ax.plot(ep, [c["trainMean"] for c in run], "--", color="#999", label="train (ε-greedy), seed1")
        if board == "12x12":
            ax.axhline(R["baselines12"]["greedy"], color="#b3261e", ls=":", label="greedy heuristic")
        ax.set(title=f"{board.replace('x', '×')} (win = {R['runs'][board + '_seed1']['winScore']})", xlabel="epoch (100 episodes)", ylabel="score")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=160)
    plt.close(fig)
    figs["curves"] = {"png": out / "curves.png", "caption": "Greedy evaluation score on 500 held-out seeds at each checkpoint, three training seeds."}

    fig, ax = plt.subplots(figsize=(7, 3))
    k = np.arange(101)
    for g, c in zip([0.5, 0.9, 0.95, 0.99], ["#999", C[2], C[0], C[1]]):
        ax.plot(k, g ** k, color=c, label=f"γ={g} (horizon ≈ {1 / (1 - g):.0f})")
    ax.set(xlabel="steps into the future k", ylabel="weight γ^k")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "gamma_horizon.png", dpi=160)
    plt.close(fig)
    figs["gamma_horizon"] = {"png": out / "gamma_horizon.png", "caption": "Discount weights and effective horizon 1/(1−γ)."}

    c = R["config"]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    e = np.arange(c["epochs"] + 1)
    ax.plot(e, np.maximum(c["epsEnd"], c["epsStart"] - (c["epsStart"] - c["epsEnd"]) * e / c["epsDecayEpochs"]), color=C[2])
    ax.set(xlabel="epoch", ylabel="ε")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "epsilon_schedule.png", dpi=160)
    plt.close(fig)
    figs["epsilon_schedule"] = {"png": out / "epsilon_schedule.png", "caption": "ε-greedy schedule used in the run."}

    sw = R["sweep"]
    means = np.array(sw["means"])
    mu, sd = means.mean(2), means.std(2, ddof=1)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    im = ax.imshow(mu, cmap="RdYlGn", vmin=10, vmax=23)
    ax.set_xticks(range(len(sw["alphas"])), [f"α={a}" for a in sw["alphas"]])
    ax.set_yticks(range(len(sw["gammas"])), [f"γ={g}" for g in sw["gammas"]])
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            ax.text(j, i, f"{mu[i, j]:.1f}±{sd[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="mean greedy score")
    fig.tight_layout()
    fig.savefig(out / "sweep_heatmap.png", dpi=160)
    plt.close(fig)
    figs["sweep_heatmap"] = {"png": out / "sweep_heatmap.png", "caption": f"γ × α sweep ({sw['note']}); mean ± sd over seeds."}

    rows = []
    for board in ["12x12", "6x6"]:
        run = R["runs"][f"{board}_seed1"]
        rows.append(f"\n**{board.replace('x', '×')}, seed 1 (win = score {run['winScore']})**\n")
        rows.append("| epoch | ε | train mean (max, wins) | eval mean | median | p90 | max | wins/500 | wall / self / timeout |")
        rows.append("|---:|---:|---|---:|---:|---:|---:|---:|---|")
        for ck in run["checkpoints"]:
            tr = "–" if ck["trainMean"] is None else f"{ck['trainMean']} ({ck['trainMax']}, {ck['trainWins']})"
            p = lambda k: f"{round(100 * ck[k] / 500)}%"
            rows.append(f"| {ck['epoch']} | {ck['eps']} | {tr} | **{ck['evalMean']}** | {ck['median']} | {ck['p90']} | {ck['max']} | {ck['wins']} | {p('wall')} / {p('self')} / {p('timeout')} |")
    figs["checkpoint_table"] = {"md": "\n".join(rows) + "\n"}
    return figs

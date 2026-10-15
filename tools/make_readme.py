# Builds README.md from README.template.md by filling {{PLACEHOLDERS}} with tables computed from the registry (models/*/meta.json),
# the run summaries (results/) and the benchmark files, so every number in the README comes from a file in this repository.
#   python tools/make_readme.py
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def metas():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "models").glob("*/meta.json"))]


def fill(t):
    return t["mean"] / (t["size"] ** 2 - 3) if "size" in t else None


def f1(x):
    return "–" if x is None else f"{x:.1f}"


def leaderboard():
    rows = ["| Agent | Family | Board | Test mean ± sd (500 games) | Fill | Max | Wins | Across training seeds | Params | CPU latency | Size |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    ms = metas()
    ms.sort(key=lambda m: (-m["size"], -(m["test"]["mean"] / (m["size"] ** 2 - 3))))
    for m in ms:
        t, ss = m["test"], m.get("seed_stats")
        cells = m["size"] ** 2 - 3
        fam = "planning (not RL)" if m.get("non_rl") else m["family"].upper() if m["family"] != "tabular" else "Tabular Q"
        seeds = f"{ss['mean']:.1f} ± {ss['std']:.1f} (n={ss['n']}; {ss['scope']})" if ss else "single run"
        lat = f"{m['onnx_cpu_latency_us']:.0f} µs" if m.get("onnx_cpu_latency_us") else "lookup"
        name = m["name"] + (" *(provisional)*" if m.get("provisional") else "")
        wins = f"{t['wins']}/{t['n']}" if t.get("wins") else "0"
        params = f"{m['params']:,}" if m.get("params") else "–"
        rows.append(f"| {name} | {fam} | {m['size']}×{m['size']} | {t['mean']:.1f} ± {t.get('std', 0):.1f} | {100 * t['mean'] / cells:.0f}% | {t['max']} | {wins} | "
                    f"{seeds} | {params} | {lat} | {m.get('json_mb', '–')} MB |")
    return "\n".join(rows)


def arms_table(path, key):
    p = ROOT / "notes" / "tryouts" / path / "results.json"
    if not p.exists():
        return "(results not available)"
    arms = json.loads(p.read_text(encoding="utf-8")).get("arms", [])
    rows = ["| Arm | Seeds | Test mean per seed | Mean ± sd (n−1) | Best game | Deaths wall / self / timeout (mean of 500) |", "|---|---:|---|---:|---:|---|"]
    for a in arms:
        m = [r["best"]["test"]["mean"] for r in a["runs"]]
        sd = statistics.stdev(m) if len(m) > 1 else 0.0
        cs = [round(statistics.mean(r["best"]["test"]["causes"].get(k, 0) for r in a["runs"])) for k in ("wall", "self", "timeout")]
        rows.append(f"| {a['label']} | {len(m)} | {', '.join(f'{x:.1f}' for x in m)} | **{statistics.mean(m):.1f} ± {sd:.1f}** | "
                    f"{max(r['best']['test']['max'] for r in a['runs'])} | {' / '.join(map(str, cs))} |")
    return "\n".join(rows)


def wins_table():
    p = ROOT / "notes" / "tryouts" / "05_win_challenge" / "results.json"
    if not p.exists():
        return "(results not available)"
    R = json.loads(p.read_text(encoding="utf-8"))
    rows = ["| Board | Cells to fill | Run | Start from | Win rate (500 held-out games) | Mean fill | Training steps |", "|---:|---:|---|---|---:|---:|---:|"]
    for st in R["stages"]:
        for r in st["runs"]:
            t = r["best"]["test"]
            rows.append(f"| {st['size']}×{st['size']} | {st['size'] ** 2} | {r.get('label', r['name'])}{' *(paused, not finished)*' if r.get('in_progress') else ''} | {r.get('start', r['init_from'] or 'scratch')} | **{100 * t['wins'] / t.get('n', 500):.1f}%** "
                        f"({t['wins']}/{t.get('n', 500)}) | {100 * t['mean'] / (st['size'] ** 2 - 3):.0f}% | {r['steps'] / 1e6:.1f} M |")
    for h in R.get("hamilton") or []:
        rows.append(f"| {h['size']}×{h['size']} | {h['size'] ** 2} | Hamiltonian cycle (planning, not RL) | – | **{100 * h['win_rate']:.0f}%** | 100% | 0 |")
    return "\n".join(rows)


def throughput_table():
    rows = ["| Workload | Jobs | Aggregate steps/s | Speed-up | GPU util | CPU util | GPU memory in use |", "|---|---:|---:|---:|---:|---:|---:|"]
    for fam in ("ppo", "dqn"):
        p = RES / "bench" / f"parallel_{fam}.json"
        if not p.exists():
            continue
        R = json.loads(p.read_text(encoding="utf-8"))
        base = R["rows"][0]["aggregate_steps_per_s"]
        for r in R["rows"]:
            rows.append(f"| {fam.upper()} | {r['jobs']} | {r['aggregate_steps_per_s']:,} | {r['aggregate_steps_per_s'] / base:.2f}× | {r['gpu_util_pct']}% | {r['cpu_util_pct']}% | {r['gpu_mem_used_mb']:,} MB |")
    return "\n".join(rows)


def training_hours():
    """Summed per-epoch training time of every logged DQN and PPO run. An epoch longer than 3x its run's median (PC sleep, a driver
    hang) is counted as 3x the median. Several runs shared the GPU, so this is training time, not exclusive GPU time."""
    total = 0.0
    for f in glob.glob(str(RES / "phase*" / "**" / "metrics.jsonl"), recursive=True):
        s = []
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "seconds" in d and "checkpoint" not in d:
                s.append(d["seconds"])
        if s:
            m = statistics.median(s)
            total += sum(min(x, 3 * m) for x in s)
    return total / 3600


def trajectory():
    """12x12 learning trajectory: table (every ~25 M steps) and a chart in docs/12x12_trajectory.png, from the two runs' checkpoint logs."""
    pts, off = [], 0
    for run in ("t05_12x12_win6_scratch", "t05_12x12_ext"):
        for line in (RES / "phase4_ppo" / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            if '"checkpoint"' in line:
                c = json.loads(line)["checkpoint"]
                pts.append((off + c["env_steps"], c["test"]["mean"] / 141, c["test"]["wins"] / c["test"]["n"]))
        off = pts[-1][0]
    rows = ["| Total steps on 12×12 | Mean fill | Win rate (500 games) |", "|---:|---:|---:|"]
    want = [1e6, 24e6, 36e6, 49e6, 74e6, 99e6, 124e6, 149e6, 158e6]
    for w in want:
        s, f, r = min(pts, key=lambda p: abs(p[0] - w))
        rows.append(f"| {s / 1e6:.0f} M | {100 * f:.0f}% | {100 * r:.1f}% |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=130)
        xs = [p[0] / 1e6 for p in pts]
        ax.plot(xs, [100 * p[2] for p in pts], color="#306230", lw=1.8, label="win rate")
        ax.plot(xs, [100 * p[1] for p in pts], color="#c27c0e", lw=1.4, label="mean fill")
        ax.axvline(24, color="#888", ls=":", lw=1)
        ax.text(24.8, 4, "extension starts", fontsize=7, color="#555")
        ax.set_xlabel("training steps on the 12×12 board (millions)")
        ax.set_ylabel("% of 500 held-out games / of the board")
        ax.set_ylim(0, 100)
        ax.grid(alpha=.25)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("PPO win agent, 12×12: greedy evaluation at every checkpoint", fontsize=9)
        fig.tight_layout()
        fig.savefig(ROOT / "docs" / "12x12_trajectory.png")
    except ImportError:
        pass
    return "\n".join(rows)


def main():
    t = (ROOT / "README.template.md").read_text(encoding="utf-8")
    for key, val in (("LEADERBOARD", leaderboard()), ("ABLATION03", arms_table("03_dqn_upgrades", "arms")), ("PPOARMS", arms_table("04_ppo", "arms")),
                     ("WINS", wins_table()), ("THROUGHPUT", throughput_table()), ("TRAJECTORY", trajectory()), ("HOURS", f"{training_hours():.0f}")):
        t = t.replace("{{" + key + "}}", val)
    (ROOT / "README.md").write_text(t, encoding="utf-8")
    print("README.md written,", len(t.splitlines()), "lines")


if __name__ == "__main__":
    main()

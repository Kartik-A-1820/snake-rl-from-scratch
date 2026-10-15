# Builds README.md from README.template.md by filling {{PLACEHOLDERS}} with tables computed from the registry (models/*/meta.json),
# the run summaries (results/) and the benchmark files, so every number in the README comes from a file in this repository.
#   python tools/make_readme.py
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


def main():
    t = (ROOT / "README.template.md").read_text(encoding="utf-8")
    for key, val in (("LEADERBOARD", leaderboard()), ("ABLATION03", arms_table("03_dqn_upgrades", "arms")), ("PPOARMS", arms_table("04_ppo", "arms")),
                     ("WINS", wins_table()), ("THROUGHPUT", throughput_table())):
        t = t.replace("{{" + key + "}}", val)
    (ROOT / "README.md").write_text(t, encoding="utf-8")
    print("README.md written,", len(t.splitlines()), "lines")


if __name__ == "__main__":
    main()

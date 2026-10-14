# Builds every learning note from one source of truth per tryout:
#   notes/tryouts/<NN_name>/content.md   theory + analysis (Markdown + LaTeX, ```widget``` and ```quiz``` blocks)
#   notes/tryouts/<NN_name>/results.json numbers from the actual run (written by the training scripts)
#   notes/tryouts/<NN_name>/widgets.js   interactive figures for the HTML version
#   notes/tryouts/<NN_name>/figures.py   static figures/tables for the DOCX version (from the same results.json)
# Outputs: notes/<NN_name>.html, notes/<NN_name>.docx, notes/snake_rl_master.docx
# Usage: python notes/build_notes.py [--only 01_tabular_q_learning] [--no-docx]
import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
TRYOUTS = NOTES / "tryouts"
BUILD = NOTES / "build"
sys.stdout.reconfigure(encoding="utf-8")


def engine_source():
    html = (ROOT / "game" / "index.html").read_text(encoding="utf-8")
    m = re.search(r"// === SNAKE CORE BEGIN ===[\s\S]*?// === SNAKE CORE END ===", html)
    return m.group(0)


def load_results(d):
    r = json.loads((d / "results.json").read_text(encoding="utf-8"))
    # "xxxFile": "path.json" pulls that file in as r["xxx"] (Q-table, benchmark results, ...)
    for k in [k for k in r if k.endswith("File")]:
        p = ROOT / r[k]
        if p.exists():
            r[k[:-4]] = json.loads(p.read_text(encoding="utf-8"))
    return r


def title_of(md):
    m = re.search(r'^title:\s*"?(.*?)"?\s*$', md, re.M)
    return m.group(1) if m else "Untitled"


def safe(s):
    return s.replace("</script", "<\\/script")


def build_html(d, md, results):
    tpl = (NOTES / "template.html").read_text(encoding="utf-8")
    widgets = (d / "widgets.js").read_text(encoding="utf-8") if (d / "widgets.js").exists() else ""
    out = (tpl.replace("%%TITLE%%", title_of(md))
              .replace("%%MD_RAW%%", safe(md))
              .replace("%%RESULTS_RAW%%", safe(json.dumps(results, separators=(",", ":"))))
              .replace("%%ENGINE%%", engine_source())
              .replace("%%WIDGETS%%", safe(widgets)))
    path = NOTES / f"{d.name}.html"
    path.write_text(out, encoding="utf-8", newline="\n")
    return path


def figures_for(d, results):
    f = d / "figures.py"
    if not f.exists():
        return {}
    spec = importlib.util.spec_from_file_location(f"fig_{d.name}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    figdir = BUILD / d.name
    figdir.mkdir(parents=True, exist_ok=True)
    return mod.make(results, figdir)


def quiz_to_md(body):
    q, opts, why = "", [], ""
    for line in body.splitlines():
        if line.startswith("Q:"):
            q = line[2:].strip()
        elif re.match(r"- \[( |x)\]", line):
            opts.append((line[6:].strip(), line[3] == "x"))
        elif line.startswith("Why:"):
            why = line[4:].strip()
    letters = "abcdefgh"
    right = ", ".join(f"({letters[i]})" for i, (_, ok) in enumerate(opts) if ok)
    lines = [f"> **Self-check.** {q}", ">"]
    lines += [f"> ({letters[i]}) {t}  " for i, (t, _) in enumerate(opts)]
    lines += [">", f"> *Answer: {right}.* {why}"]
    return "\n".join(lines)


def docx_markdown(md, figs):
    def widget(m):
        name = m.group(1).strip()
        f = figs.get(name)
        if not f:
            return f"*[Interactive figure “{name}”: open the HTML version of this note.]*"
        if "md" in f:
            return f["md"]
        return f"![{f['caption']}]({Path(f['png']).as_posix()}){{width=100%}}"
    md = re.sub(r"```widget\n([\s\S]*?)```", widget, md)
    return re.sub(r"```quiz\n([\s\S]*?)```", lambda m: quiz_to_md(m.group(1)), md)


def pandoc(src, dst, extra=()):
    exe = shutil.which("pandoc")
    if not exe:
        print("  ! pandoc not found, skipping", dst.name, "(install: winget install --id JohnMacFarlane.Pandoc)")
        return False
    ref = NOTES / "reference.docx"
    cmd = [exe, str(src), "-f", "markdown", "-o", str(dst), "--resource-path", str(BUILD), *extra]
    if ref.exists():
        cmd += ["--reference-doc", str(ref)]
    subprocess.run(cmd, check=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--no-docx", action="store_true")
    a = ap.parse_args()
    BUILD.mkdir(exist_ok=True)
    chapters = []
    for d in sorted(p for p in TRYOUTS.iterdir() if p.is_dir()):
        if not (d / "results.json").exists():
            print(f"- {d.name}: no results.json yet (run the experiment first), skipped")
            continue
        md = (d / "content.md").read_text(encoding="utf-8")
        results = load_results(d)
        print(f"- {d.name}")
        print("  html :", build_html(d, md, results).relative_to(ROOT))
        figs = figures_for(d, results)
        dmd = docx_markdown(md, figs)
        src = BUILD / f"{d.name}.md"
        src.write_text(dmd, encoding="utf-8")
        chapters.append((title_of(md), dmd))
        if not a.no_docx and (a.only in (None, d.name)):
            dst = NOTES / f"{d.name}.docx"
            if pandoc(src, dst, ["--toc", "--toc-depth=2"]):
                print("  docx :", dst.relative_to(ROOT))

    if chapters and not a.no_docx:
        parts = ['---\ntitle: "Snake RL: learning notes"\nsubtitle: "Reinforcement learning from tabular Q to deep RL, one tryout per chapter"\n---\n']
        for title, dmd in chapters:
            body = re.sub(r"^---\n[\s\S]*?\n---\n", "", dmd)
            body = re.sub(r"^(#+) ", r"#\1 ", body, flags=re.M)  # each tryout becomes one chapter
            parts.append(f"\n# {title}\n\n{body}\n")
        master = BUILD / "snake_rl_master.md"
        master.write_text("\n".join(parts), encoding="utf-8")
        dst = NOTES / "snake_rl_master.docx"
        if pandoc(master, dst, ["--toc", "--toc-depth=2"]):
            print("master:", dst.relative_to(ROOT))


if __name__ == "__main__":
    main()

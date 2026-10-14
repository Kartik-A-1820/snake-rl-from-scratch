# Contributing

Thanks for looking. This is a teaching repository, so small, clear contributions matter most.

**Good first contributions**
- A typo, a broken link, or an unclear sentence in a note (`notes/tryouts/NN_name/content.md`, then `python notes/build_notes.py`).
- A new tryout (another algorithm, or an ablation of an existing one). Copy the structure of `notes/tryouts/03_dqn_upgrades/`: `content.md` (theory and analysis), `widgets.js` (interactive figures), `figures.py` (static versions for the .docx), and a results file written by a script.
- A reproduction on other hardware (add a row to the hardware table in the README).

**Ground rules**
- Run `powershell -ExecutionPolicy Bypass -File .\run.ps1 test` before opening a pull request. The Python and JavaScript engines must stay bit-identical; `python/tests/test_parity.py` checks that.
- Report results the way the existing notes do: 500 held-out seeds, a validation set for choosing a checkpoint, several training seeds, mean ± standard deviation. Say plainly when a result is one seed.
- Keep the game a single file that works when double-clicked (no server, no `fetch()` of local files).
- Do not commit raw run checkpoints (`models/_runs/`) or large binaries. The best model per algorithm is published to `models/<algo>/` by `python/export_model.py --algo ...`.

Questions and ideas are welcome as issues.

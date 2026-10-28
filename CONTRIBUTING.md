# Contributing

Thanks for looking. This is a teaching repository, so small, clear contributions matter most.

**Good first contributions**

- A typo, a broken link, or an unclear sentence in a note (`notes/tryouts/NN_name/content.md`, then `python notes/build_notes.py`).
- A new tryout (another algorithm, or an ablation of an existing one). Copy the structure of `notes/tryouts/03_dqn_upgrades/`: `content.md` (theory and analysis), `widgets.js` (interactive figures), `figures.py` (static versions for the .docx), and a results file written by a script.
- A reproduction on other hardware, with raw results and environment details. Update generated result tables through their source records.

**Ground rules**

- Start with the CPU check in [README.md](README.md), then run the applicable parity checks. On Windows, run `powershell -ExecutionPolicy Bypass -File .\run.ps1 test` before opening a pull request. The Python and JavaScript engines must stay bit-identical; `python/tests/test_parity.py` checks that.
- Report results the way the existing notes do: 500 held-out seeds, a validation set for choosing a checkpoint, several training seeds, mean ± standard deviation. Say plainly when a result is one seed.
- Keep the game a single file that works when double-clicked (no server, no `fetch()` of local files).
- Do not commit raw run checkpoints (`models/_runs/`) or large binaries. The best model per algorithm is published to `models/<algo>/` by `python/export_model.py --algo ...`.

Questions and ideas are welcome as issues.

## Documentation and experiment changes

Edit `README.template.md` before regenerating `README.md` with `python tools/make_readme.py`. Keep published records and labels for single-run versus multi-seed results. Do not type a new result into a generated table.

For a pull request, describe the change and list the commands run, outcomes, environment, and untested paths. New algorithm comparisons should use the same board, observation, training budget, validation-selection policy, and test seeds. Include unsuccessful configurations when they affect the conclusion.

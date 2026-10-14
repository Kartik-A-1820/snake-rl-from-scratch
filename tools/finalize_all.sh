#!/bin/bash
# One command to refresh every generated artifact from the current results (run again after any training run finishes):
#   results -> notes data -> notes (HTML + DOCX + master book) -> README tables -> hero GIFs -> agent parity data
# Needs the CUDA venv (.venv-cuda), pandoc on PATH for the DOCX files, and nothing else heavy running on the GPU for the GIF step.
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONPATH=python
PY=./.venv-cuda/Scripts/python.exe
$PY python/make_notes_data.py
$PY notes/build_notes.py
$PY tools/make_readme.py
$PY tools/make_gif.py --out docs/agents.gif --agents 01_tabular_q 02_dqn_cnn 04_ppo --greedy
$PY tools/make_gif.py --out docs/wins.gif --agents 05_ppo_win_6x6 05_ppo_win_8x8 05_ppo_win_10x10 05_hamiltonian --seed-search
$PY tools/agent_parity.py

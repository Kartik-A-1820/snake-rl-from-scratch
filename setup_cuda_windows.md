# CUDA setup on Windows (GTX 1650 Ti, 4 GB)

You already have CUDA and PyTorch installed, so the plan is to **check, not install**. Everything new goes on D:.

## 1. Record what is installed

```powershell
cd path\to\snake-rl
powershell -ExecutionPolicy Bypass -File .\run.ps1 check
```

This writes `ENVIRONMENT.md` with:

- `nvidia-smi` output (driver version, and the highest CUDA version that driver supports)
- the Python version and path
- `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()`
- the GPU name, VRAM and compute capability (sm)
- a small matmul on the GPU, to confirm it actually runs
- package, node and pandoc versions

It installs nothing.

How to read `ENVIRONMENT.md`:

| `torch.version.cuda` | `cuda_available` | Meaning |
|---|---|---|
| `13.0` / `12.6` / … | True | Ready. Nothing to install. |
| `None` | False | CPU-only torch. The report prints the exact wheel command; run it only if you agree (~2.5 GB). |
| set | False | Driver/runtime mismatch, or Windows is using the integrated GPU. See the pitfalls below. |

## 2. Which CUDA wheel fits a GTX 1650 Ti (only needed if torch is CPU-only)

The GTX 1650 Ti is Turing, compute capability 7.5. In the PyTorch 2.12 support matrix (April 2026), Turing 7.5 is covered by:

- **CUDA 13.0**, the stable build. It needs a driver whose `nvidia-smi` shows "CUDA Version ≥ 13.0" (the R580 driver branch or newer):
  `python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130`
- **CUDA 12.6**, the legacy build, for older drivers:
  `python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126`

CUDA 12.8 wheels were retired in April 2026. Plain `pip install torch` on Windows gives a CPU-only build, so the `--index-url` is what matters.

To verify:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You don't need a separate CUDA Toolkit. The wheels ship their own CUDA runtime; you only need the NVIDIA driver.

## 3. Small dependencies, installed on D:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 setup
```

This creates `.venv` inside the repository with `--system-site-packages`. The venv reuses your existing CUDA torch, and everything new (gymnasium, onnx, onnxruntime, matplotlib, pytest) installs under D:. The pip cache also lives on D: (`.cache\pip`).

## Pitfalls on a laptop

- **Two Pythons.** If `python` on PATH isn't the one that has CUDA torch, `check` will show it (look at `python_exe`). Run `py -0p` to list them.
- **Optimus / hybrid graphics.** Plug in the charger and set Windows Graphics settings → python.exe → High performance. Training on battery is throttled.
- **4 GB VRAM.** The model is tiny (~0.3 M parameters, ~1.3 MB). What uses VRAM is the replay buffer: about 314 MB at 200k transitions stored as uint8. The trainer prints the real figure.
- **No tensor cores on TU117.** fp16 autocast gives little here, so the training code stays in fp32.
- **Small nets are launch-bound.** At batch size 1–64 a 5K-parameter MLP is often faster on the CPU. `run.ps1 bench` measures this on your machine, so don't take it on faith.

Sources: [PyTorch: CUDA 13.2 / deprecating 12.8 (Release 2.12)](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337), [PyTorch: CUDA 13.0 as stable (Release 2.11)](https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325), [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

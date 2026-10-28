# Windows environment setup

The published experiments used a GTX 1650 Ti laptop. That is a reference environment; it is not a requirement for the browser demo or CPU environment tests.

## Inspect the current environment

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 check
```

This writes `ENVIRONMENT.md` with the interpreter, PyTorch/CUDA availability, GPU details when available, and environment diagnostics. It does not install packages.

## Choose CPU tests or GPU training

For CPU checks, follow the isolated-environment commands in [README.md](README.md). Requirements intentionally omit torch so readers can select the appropriate build.

For GPU work, choose an installation command for your operating system, Python version, GPU, and driver using the [official PyTorch installer](https://pytorch.org/get-started/locally/). Check that the selected build supports your GPU rather than relying only on a version printed by `nvidia-smi`.

After installation, inspect:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A successful availability check is a starting point; validate the actual model/workload before a long training run.

## Convenience script behavior

`run.ps1 setup` creates a repository-local `.venv` with `--system-site-packages`, reusing visible global packages. If an NVIDIA GPU is detected and CUDA torch is missing, it selects one of its hardcoded CUDA wheel indexes from the driver report and can install a large package.

This heuristic is not a compatibility guarantee. For controlled experiments, create an isolated environment and install a build you have checked explicitly. The script's `-NoCuda` option skips automatic CUDA installation; it does not install CPU torch for you.

`run.ps1 test` generates JavaScript traces using Node.js and runs Python tests. Record the interpreter, package versions, GPU/driver, and test results when reporting a reproduction. Training and device benchmarks are separate from correctness checks.

## Common setup issues

- Check that the active `python` is the interpreter where PyTorch is installed.
- Use a separate environment for CPU checks and CUDA training.
- On laptops, power/thermal conditions and background workloads affect throughput.
- Do not copy another machine's drive paths or assume available VRAM determines job throughput.

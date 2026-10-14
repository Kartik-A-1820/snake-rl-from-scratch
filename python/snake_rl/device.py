import platform
import shutil
import sys

import torch


def _cuda_unavailable_reason():
    if torch.version.cuda is None:
        return (f"this torch ({torch.__version__}) is a CPU-only build. Install the CUDA wheel, e.g. "
                "`python -m pip install torch --index-url https://download.pytorch.org/whl/cu130` "
                "(cu126 for drivers older than CUDA 13), or run `.\run.ps1 setup`.")
    return (f"torch {torch.__version__} has CUDA {torch.version.cuda} but cuda.is_available() is False "
            "(driver too old for this wheel, or Windows is using the integrated GPU). Check `nvidia-smi`.")


def pick_device(want="auto"):
    if want == "auto":
        if torch.cuda.is_available():
            want = "cuda"
        else:
            want = "cpu"
            if shutil.which("nvidia-smi"):  # an NVIDIA GPU is present but unusable: don't fall back silently
                bar = "!" * 78
                print(bar, file=sys.stderr)
                print(f"[device] WARNING: NVIDIA GPU detected but training on CPU: {_cuda_unavailable_reason()}", file=sys.stderr)
                print("[device] Pass --device cpu to silence this.", file=sys.stderr)
                print(bar, file=sys.stderr, flush=True)
    elif want.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"[device] --device {want} requested but CUDA is unavailable: {_cuda_unavailable_reason()}")
    dev = torch.device(want)
    print(f"[device] {describe(dev)}")
    return dev


def describe(dev=None):
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        return (f"cuda:{dev.index or 0} {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}, "
                f"torch {torch.__version__} (CUDA {torch.version.cuda})")
    return f"cpu {platform.processor() or platform.machine()}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def sync(dev):
    # CUDA kernels are async; without this, timings measure launch, not work
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

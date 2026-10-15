# Removes the debugging metadata torch.onnx adds to every node (stack traces with local file paths) from the exported ONNX files.
#   python tools/strip_onnx_meta.py [files...]   (default: models/*/best.onnx)
import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]


def strip(path):
    m = onnx.load(str(path), load_external_data=False)
    for n in m.graph.node:
        del n.metadata_props[:]
    del m.metadata_props[:]
    onnx.save(m, str(path))


if __name__ == "__main__":
    for f in (sys.argv[1:] or sorted(str(p) for p in (ROOT / "models").glob("*/best.onnx"))):
        strip(f)
        print("stripped", f)

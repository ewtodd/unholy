"""Cross-check unholy's EXL3 CPU decode against exllamav3's own reconstruction.

    PYTHONPATH=/scratch/unholy-ref/exllamav3:/scratch/unholy-ref/exllamav3/.stubs \
        $REF_PY tools/ref/exl3_ref.py <ckpt_dir> <prefix> <ours_inner.f16> <ours_dense.f32>

Inner weights must match bit for bit. The dense weight is compared against exllamav3's
get_weight_tensor(), which rounds to fp16 after each step, so it is checked by relative error.
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

from exllamav3.modules.quant.exl3 import LinearEXL3


def load(ckpt, prefix, device):
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    out = {}
    for suffix in ("trellis", "suh", "svh", "mul1", "mcg"):
        key = f"{prefix}.{suffix}"
        if key in wm:
            with safe_open(os.path.join(ckpt, wm[key]), "pt", device="cpu") as f:
                out[suffix] = f.get_tensor(key).to(device)
    return out


def main():
    ckpt, prefix, inner_path, dense_path = sys.argv[1:5]
    dev = torch.device("cuda:0")
    t = load(ckpt, prefix, dev)
    k = t["trellis"].shape[0] * 16
    n = t["trellis"].shape[1] * 16
    lin = LinearEXL3(None, k, n, suh=t["suh"], svh=t["svh"], trellis=t["trellis"],
                     mul1=t.get("mul1"), mcg=t.get("mcg"), key=prefix)
    ref_inner = lin.get_inner_weight_tensor().cpu().numpy().view(np.uint16)
    ref_dense = lin.get_weight_tensor().float().cpu().numpy()

    ours_inner = np.fromfile(inner_path, dtype=np.uint16).reshape(k, n)
    ours_dense = np.fromfile(dense_path, dtype=np.float32).reshape(k, n)

    mism = int((ref_inner != ours_inner).sum())
    print(f"{prefix}: [{k}, {n}] inner mismatches {mism} / {k * n}")
    if mism:
        idx = np.argwhere(ref_inner != ours_inner)[:5]
        for r, c in idx:
            print(f"  ({r},{c}) ref {ref_inner[r, c]:04x} ours {ours_inner[r, c]:04x}")
    diff = np.abs(ref_dense - ours_dense)
    scale = np.abs(ours_dense).max()
    rel = float(np.linalg.norm(ref_dense - ours_dense) / np.linalg.norm(ours_dense))
    print(f"  dense: max|d| {diff.max():.3e} (max|w| {scale:.3e}), rel L2 {rel:.3e}")
    sys.exit(0 if mism == 0 and rel < 2e-3 else 1)


if __name__ == "__main__":
    main()

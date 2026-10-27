"""GLM-5.3-Flash's vision tower at the release's precision, beside an EXL3 quant of the language model.

    $GLM_PY tools/box/extract-glm-vision.py --release /scratch/models/glm53-flash-fp8 \
        --out /scratch/models/glm53-flash-exl3-4.05bpw/vision.safetensors

The FP8 release keeps the ViT and its merger in bf16 (~1.1 GB); exllamav3's quant re-quantizes their linears to
EXL3. unholy serves the release's: this copies every model.visual.* tensor, unchanged, into one safetensors file
under vision.* (the quant's own index already uses the model.visual.* names, and a side file's tensors join it),
with the source recorded in the metadata. `unholy serve` loads it when it is there.
"""
import argparse
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    index = json.load(open(os.path.join(a.release, "model.safetensors.index.json")))["weight_map"]
    by_file = {}
    for name, f in index.items():
        if name.startswith("model.visual."):
            by_file.setdefault(f, []).append(name)
    tensors = {}
    for f, names in sorted(by_file.items()):
        with safe_open(os.path.join(a.release, f), framework="pt") as sf:
            for name in names:
                tensors["vision." + name[len("model.visual."):]] = sf.get_tensor(name).contiguous()
    meta = {"source": os.path.abspath(a.release), "renamed": "model.visual.* -> vision.*"}
    tmp = a.out + ".tmp"
    save_file(tensors, tmp, metadata=meta)
    os.replace(tmp, a.out)
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    dtypes = sorted({str(t.dtype) for t in tensors.values()})
    print(f"{a.out}: {len(tensors)} tensors, {total / 1e9:.2f} GB, {dtypes}")


if __name__ == "__main__":
    main()

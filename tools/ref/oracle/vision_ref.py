"""Reference image-block embeddings: DeepSeek's inference/vision.py (ViT + Aligner) and image_processor.py, as
model.py merge_image_embeddings builds them (learned rows by token type, aligner rows in the N-layout).

    $REF_PY tools/ref/oracle/vision_ref.py --ckpt /scratch/dsv4-ve --image FILE --out FILE [--device cpu]

Writes bf16 [block_len, dim] (the block for start position 0) to --out and prints the grid.
"""
import argparse
import json
import os
import struct
import sys
from types import SimpleNamespace

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/dsv4-ve")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.ckpt, "inference"))
    import image_processor as ip
    import vision as V

    with open(os.path.join(a.ckpt, "inference", "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(a.ckpt, "config.json")) as f:  # vision fields the inference config leaves to defaults
        for k, v in json.load(f).items():
            if k.startswith("vision_"):
                cfg.setdefault(k, v)
    args = SimpleNamespace(**cfg)
    torch.set_default_dtype(torch.bfloat16)
    dev = torch.device(a.device)

    wm = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
    hdrs = {}

    def get(name):
        fn = wm[name]
        if fn not in hdrs:
            with open(os.path.join(a.ckpt, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdrs[fn] = (8 + n, json.loads(f.read(n)))
        off, h = hdrs[fn]
        e = h[name]
        b0, b1 = e["data_offsets"]
        with open(os.path.join(a.ckpt, fn), "rb") as f:
            f.seek(off + b0)
            raw = f.read(b1 - b0)
        assert e["dtype"] == "BF16", (name, e["dtype"])
        return torch.from_numpy(np.frombuffer(raw, dtype=np.int16).copy()).view(torch.bfloat16).reshape(e["shape"])

    with torch.no_grad():
        vit = V.ViT(args).to(dev)
        al = V.Aligner(args).to(dev)
        for mod, prefix in ((vit, "vision."), (al, "aligner.")):
            for pname, p in mod.named_parameters():
                p.data.copy_(get(prefix + pname).to(dev).to(p.dtype))
        params = torch.stack([get(n) for n in ("image_start", "image_pad", "image_pad", "image_newline", "image_end")]).to(dev)
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"url": a.image}, args)
        types, perm = ip.build_image_block(n_llm_h, n_llm_w, 0)
        embeds = al(vit(patches.to(dev), n_vit_h, n_vit_w), n_vit_h, n_vit_w)[perm.to(dev)]
        block = params[types.to(dev)].clone()
        block[types.to(dev) == ip.IMAGE] = embeds
    block.float().to(torch.bfloat16).cpu().contiguous().view(torch.int16).numpy().tofile(a.out)
    print(f"grid {n_vit_h}x{n_vit_w} -> {n_llm_h}x{n_llm_w}, block {block.shape[0]} rows, |block| {block.float().norm():.4e}")


if __name__ == "__main__":
    main()

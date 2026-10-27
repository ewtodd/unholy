"""GLM-5.3-Flash's vision tower: unholy-vit-test (model/glm5_vision) vs transformers' Glm5NextVisionModel.

    $GLM_PY tools/ref/glm_vit_compare.py --tool build/unholy-vit-test \
        --model /scratch/models/glm53-flash-exl3-4.05bpw [--dev 2] [--keep DIR]

The reference loads the same vision.safetensors on cuda:0 and runs each image twice: in bf16 (as served) and in
fp32 on the same bf16 patches (the truth both are measured against; skipped for images whose fp32 attention would
not fit). Prints, per image, the relative L2 error against fp32 of ours and of the bf16 reference, and the share of
values ours has bit-identical with the bf16 reference. Fails when ours is more than twice the bf16 reference's error.
"""
import argparse
import io
import os
import random
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_compare import encode, synth  # noqa: E402


def scene(rng, w, h):
    """Flat shapes and text: closer to a photo or a screenshot than noise."""
    img = Image.new("RGB", (w, h), tuple(rng.randrange(256) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x0, y0 = rng.randrange(w), rng.randrange(h)
        box = [x0, y0, x0 + rng.randrange(w // 3 + 1), y0 + rng.randrange(h // 3 + 1)]
        (d.ellipse if rng.random() < 0.5 else d.rectangle)(box, fill=tuple(rng.randrange(256) for _ in range(3)))
    for _ in range(12):
        d.text((rng.randrange(w), rng.randrange(h)), "unholy GLM vision 0123",
               fill=tuple(rng.randrange(256) for _ in range(3)))
    return img


def build(cfg, sd, dtype):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    # Built under the default dtype, as from_pretrained does, so the rope's fp32 inv_freq stays fp32.
    torch.set_default_dtype(dtype)
    try:
        with torch.device("cuda:0"):
            m = Glm5NextVisionModel(cfg)
    finally:
        torch.set_default_dtype(torch.float32)
    m.load_state_dict(sd, strict=True)
    return m.eval()


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def worst_row(a, b):
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--fp32-max-patches", type=int, default=16000)
    ap.add_argument("--keep", help="directory for the images and both sides' rows")
    a = ap.parse_args()
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorPil

    cfg = AutoConfig.from_pretrained(a.model).vision_config
    cfg._attn_implementation = "sdpa"
    sd = {k[len("vision."):]: v for k, v in load_file(os.path.join(a.model, "vision.safetensors")).items()}
    m16, m32 = build(cfg, sd, torch.bfloat16), build(cfg, sd, torch.float32)
    del sd
    proc = Glm5NextImageProcessorPil(max_image_tokens=a.max_tokens)

    rng = random.Random(a.seed)
    cases = [("scene", 28, 28), ("scene", 448, 448), ("noise", 640, 480), ("scene", 1024, 768),
             ("scene", 300, 2000), ("scene", 1920, 1080), ("noise", 1280, 1280), ("scene", 4032, 3024)]
    work = a.keep or tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    bad = 0
    for i, (kind, w, h) in enumerate(cases):
        name = f"{i}-{kind}-{w}x{h}"
        img = scene(rng, w, h) if kind == "scene" else synth(rng, w, h, "RGB")
        data = encode(img, rng.choice(["PNG", "JPEG"]), rng)
        path, out = os.path.join(work, name + ".img"), os.path.join(work, name + ".ours")
        with open(path, "wb") as f:
            f.write(data)
        r = subprocess.run([a.tool, "--model", a.model, "--image", path, "--out", out, "--dev", str(a.dev),
                            "--reps", "2", "--max-tokens", str(a.max_tokens)], capture_output=True, text=True)
        if r.returncode:
            print(f"{name}: ours failed: {r.stderr.strip()}")
            bad += 1
            continue
        ms = re.search(r"encode ([\d.]+) ms", r.stdout).group(1)

        ref = proc(images=[Image.open(io.BytesIO(data)).convert("RGB")], return_tensors="pt")
        pv = ref["pixel_values"].to("cuda:0").to(torch.bfloat16)
        grid = ref["image_grid_thw"].to("cuda:0")
        patches = int(grid.prod())
        try:
            with torch.no_grad():
                r16 = m16(pv, grid_thw=grid).pooler_output.float().cpu()
                r32 = m32(pv.float(), grid_thw=grid).pooler_output.cpu() if patches <= a.fp32_max_patches else None
        except torch.OutOfMemoryError:
            print(f"{name}: {patches} patches, encode {ms} ms: the reference ran out of memory", flush=True)
            continue
        finally:
            torch.cuda.empty_cache()
        ours16 = torch.from_numpy(np.fromfile(out, dtype=np.int16)).view(torch.bfloat16).view(-1, r16.shape[-1])
        ours = ours16.float()
        if ours.shape != r16.shape:
            print(f"{name}: ours {tuple(ours.shape)} vs reference {tuple(r16.shape)}")
            bad += 1
            continue
        same = (ours16.view(torch.int16) == r16.to(torch.bfloat16).view(torch.int16)).float().mean().item()
        line = f"{name}: {int(grid[0, 1])}x{int(grid[0, 2])} patches, {r16.shape[0]} tokens, encode {ms} ms: "
        if r32 is not None:
            e_ours, e_ref = rel(ours, r32), rel(r16, r32)
            line += (f"vs fp32: ours {e_ours:.2e} (worst row {worst_row(ours, r32):.2e}), bf16 reference {e_ref:.2e} "
                     f"(worst row {worst_row(r16, r32):.2e}); ")
            if e_ours > 2 * e_ref:
                line += "TOO FAR; "
                bad += 1
        line += f"vs bf16 reference {rel(ours, r16):.2e}, {same:.1%} identical"
        print(line, flush=True)
        if a.keep:
            r16.to(torch.bfloat16).view(torch.int16).numpy().tofile(os.path.join(work, name + ".ref16"))
    print(f"{len(cases)} images: {'all within 2x of the bf16 reference' if bad == 0 else f'{bad} failures'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

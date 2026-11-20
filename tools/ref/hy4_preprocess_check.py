"""Hy4-flash image preprocessing: unholy-hy4-preprocess vs transformers' HYV4VLImageProcessor, bit for bit.

    /work/env/hy4/bin/python tools/ref/hy4_preprocess_check.py --tool build/unholy-hy4-preprocess \
        --processor-config /scratch/models/hy4-flash-fp8/processor_config.json --out /work/runs/2026-11-20-hy4/preprocess

Generates the test images (sizes over/under the pixel budget, odd aspect ratios, RGBA / grey / palette inputs, an image
that needs no resize), runs both sides on the same bytes and compares the grid and token count, the resized uint8 image
(against torchvision's resize with the size transformers' smart_resize picks) and the bf16 pixel_values rows (the
processor's float32 output as the vision tower casts it). Also prints the mean / std the processor applies (Q1).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image


def make_rgb(rng, w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1), ((xx + yy) * 7) % 256], -1)
    noise = rng.integers(0, 256, size=(h, w, 3))
    mask = rng.random((h, w, 1)) < 0.3  # noisy patches and smooth ones: both the clamped and the plain paths
    return np.where(mask, noise, base).astype(np.uint8)


def cases(out):
    rng = np.random.default_rng(1)
    items = []

    def add(name, im, fmt):
        path = os.path.join(out, f"{name}.{fmt.lower()}")
        im.save(path, fmt, **({"quality": 90} if fmt == "JPEG" else {}))
        items.append((name, path))

    for h, w in [(512, 512), (500, 1000), (1000, 500), (10, 10), (3000, 2500), (333, 701), (1, 200), (64, 4000),
                 (4097, 31), (2048, 2048), (96, 96), (700, 33), (45, 45)]:
        add(f"rgb-{h}x{w}", Image.fromarray(make_rgb(rng, w, h)), "PNG")
    rgb = make_rgb(rng, 640, 480)
    rgba = Image.fromarray(rgb).convert("RGBA")
    rgba.putalpha(Image.fromarray((np.arange(640 * 480).reshape(480, 640) % 256).astype(np.uint8)))
    add("rgba-480x640", rgba, "PNG")
    add("grey-jpeg-300x400", Image.fromarray(make_rgb(rng, 400, 300)).convert("L"), "JPEG")
    add("palette-257x389", Image.fromarray(make_rgb(rng, 389, 257)).quantize(64), "PNG")
    add("noresize-256x384", Image.fromarray(make_rgb(rng, 384, 256)), "PNG")  # multiples of 32, in budget
    add("jpeg-777x555", Image.fromarray(make_rgb(rng, 555, 777)), "JPEG")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--processor-config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torchvision.transforms.v2.functional as tvF
    from transformers.models.hy_v4_vl.image_processing_hy_v4_vl import HYV4VLImageProcessor, smart_resize

    config = json.load(open(a.processor_config))["image_processor"]
    config = {k: v for k, v in config.items() if k not in ("image_processor_type",)}
    proc = HYV4VLImageProcessor(**config)
    print("processor image_mean", proc.image_mean, "image_std", proc.image_std, "rescale_factor", proc.rescale_factor,
          "size", proc.size)

    bad = 0
    for name, path in cases(a.out):
        prefix = os.path.join(a.out, name)
        res = subprocess.run([a.tool, path, prefix], capture_output=True, text=True)
        line = res.stdout.strip()
        got = json.loads(line) if line else {"error": res.stderr}
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im)
        try:
            feat = proc(images=im, return_tensors="pt")
        except ValueError as e:
            print(f"{name}: reference error {e}; tool: {got}")
            bad += "error" not in got
            continue
        thw = feat["image_grid_thw"][0].tolist()
        pv = feat["pixel_values"]
        want_tokens = (thw[1] // 2) * (thw[2] // 2 + 1)
        rh, rw = smart_resize(arr.shape[0], arr.shape[1], factor=32, min_pixels=proc.size.shortest_edge,
                              max_pixels=proc.size.longest_edge)
        ref_u8 = tvF.resize(torch.from_numpy(arr).permute(2, 0, 1), [rh, rw],
                            interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        ref_u8 = ref_u8.permute(1, 2, 0).contiguous().numpy()
        if "error" in got:
            print(f"{name}: TOOL ERROR {got['error']}")
            bad += 1
            continue
        ok_grid = [got["grid_h"], got["grid_w"], got["tokens"]] == [thw[1], thw[2], want_tokens] and thw[0] == 1
        mine = np.fromfile(prefix + ".rgb", dtype=np.uint8)
        ok_rgb = mine.size == ref_u8.size and np.array_equal(mine, ref_u8.reshape(-1))
        ndiff = int((mine != ref_u8.reshape(-1)).sum()) if mine.size == ref_u8.size else -1
        maxdiff = int(np.abs(mine.astype(int) - ref_u8.reshape(-1).astype(int)).max()) if mine.size == ref_u8.size else -1
        bf = np.fromfile(prefix + ".bf16", dtype=np.uint16)
        want = pv.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype(np.uint16).reshape(-1)
        ok_pv = bf.size == want.size and np.array_equal(bf, want)
        ok = ok_grid and ok_rgb and ok_pv
        bad += not ok
        print(f"{name:22s} {arr.shape[0]}x{arr.shape[1]} -> {rh}x{rw} grid {thw[1]}x{thw[2]} tokens {want_tokens}: "
              f"grid {'ok' if ok_grid else 'DIFF'}  resized {'ok' if ok_rgb else f'DIFF ({ndiff} bytes, max {maxdiff})'}  "
              f"pixel_values {'ok' if ok_pv else 'DIFF'}")
    print("ALL BIT-EXACT" if bad == 0 else f"{bad} case(s) differ")
    sys.exit(1 if bad else 0)


main()

"""GLM-5.3-Flash image preprocessing: unholy-vision-test glm vs transformers' Glm5NextImageProcessorPil, bit for bit.

    $GLM_PY tools/ref/glm_vision_compare.py --tool build/unholy-vision-test [--n 60]

Generates images of many sizes and aspect ratios (under the 16-token minimum, over the 8000-token budget, thin
strips), PIL modes and formats (vision_compare.py's generators), runs both sides on the same bytes and compares the
grids and the bf16 patches (the processor's float32 pixel_values as the model casts them).
"""
import argparse
import io
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_compare import encode, synth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=8000)
    a = ap.parse_args()
    from transformers.models.glm5_next.image_processing_pil_glm5_next import Glm5NextImageProcessorPil

    proc = Glm5NextImageProcessorPil(max_image_tokens=a.max_tokens)
    rng = random.Random(a.seed)
    sizes = [(1, 1), (8, 8), (13, 900), (900, 13), (2000, 200), (200, 2000), (64, 64), (383, 385), (1920, 1080),
             (1080, 1920), (4032, 3024), (8000, 6000), (500, 333), (12, 3000), (3000, 12), (640, 480), (2560, 1600),
             (28, 28), (56, 84), (100, 50)]
    modes = ["RGB", "RGB", "RGBA", "L", "LA", "P", "I;16"]
    fmts = ["JPEG", "PNG", "WEBP"]
    cases = []
    while len(cases) < a.n:
        w, h = sizes[len(cases)] if len(cases) < len(sizes) else (rng.randint(8, 3000), rng.randint(8, 3000))
        mode, fmt = rng.choice(modes), rng.choice(fmts)
        if fmt == "JPEG" and mode in ("RGBA", "LA", "P", "I;16"):
            mode = "RGB"
        if fmt == "WEBP" and mode == "I;16":
            mode = "RGB"
        try:
            data = encode(synth(rng, w, h, mode), fmt, rng)
        except Exception as e:  # PIL cannot write some combinations
            print(f"skip {w}x{h} {mode} {fmt}: {e}")
            continue
        cases.append((f"{w}x{h}-{mode}-{fmt}", data))

    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for name, data in cases:
            path, out = os.path.join(td, "img"), os.path.join(td, "patches")
            with open(path, "wb") as f:
                f.write(data)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            ref = proc(images=[img], return_tensors="np")
            grid = [int(v) for v in ref["image_grid_thw"][0]]
            pv = torch.from_numpy(np.ascontiguousarray(ref["pixel_values"])).to(torch.bfloat16)
            ref16 = pv.view(torch.int16).numpy().astype(np.uint16).reshape(-1)
            r = subprocess.run([a.tool, "glm", path, out, str(a.max_tokens)], capture_output=True, text=True)
            got = json.loads(r.stdout.strip().splitlines()[-1])
            if "error" in got:
                print(f"{name}: ours failed: {got['error']}")
                bad += 1
                continue
            ours = np.fromfile(out, dtype=np.uint16)
            ok_grid = [1, got["grid_h"], got["grid_w"]] == grid
            same = ours.shape == ref16.shape and bool((ours == ref16).all())
            diff = "" if same or ours.shape != ref16.shape else f", {int((ours != ref16).sum())} of {ref16.size} differ"
            print(f"{name}: grid {grid[1]}x{grid[2]} ({grid[1] * grid[2] // 4} tokens): grid {'ok' if ok_grid else 'DIFFERS'}, "
                  f"patches {'identical' if same else 'DIFFER' + diff}")
            bad += (not ok_grid) + (not same)
    print(f"{len(cases)} images: {'all identical' if bad == 0 else f'{bad} mismatches'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

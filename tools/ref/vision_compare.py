"""Image preprocessing: unholy-vision-test vs DeepSeek's inference/image_processor.py (PIL), bit for bit.

    $REF_PY tools/ref/vision_compare.py --tool build/unholy-vision-test --ckpt /scratch/dsv4-ve [--n 60]

Generates images of many sizes, aspect ratios (including the > 8:1 resize branch and the min-pixel
upscale), PIL modes (RGB, RGBA, L, LA, P, 16-bit grey / RGB) and formats (JPEG at several qualities and
subsamplings, PNG, WebP lossy and lossless), plus the checkpoint's example photos; runs both sides; and
compares the bf16 patches (bitwise), the grid sizes, and the token block (types, permutation) at a few
start positions.
"""
import argparse
import io
import json
import os
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image


def synth(rng, w, h, mode):
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)), ((xx + yy) * 7) % 256], -1)
    noise = np.array([rng.randrange(256) for _ in range(min(w * h * 3, 300000))], dtype=np.int64)
    flat = base.reshape(-1)
    flat[: len(noise)] = (flat[: len(noise)] + noise) % 256
    rgb = Image.fromarray(flat.reshape(h, w, 3).astype(np.uint8), "RGB")
    if mode == "RGB":
        return rgb
    if mode == "RGBA":
        a = Image.fromarray((np.arange(w * h).reshape(h, w) % 256).astype(np.uint8), "L")
        im = rgb.copy()
        im.putalpha(a)
        return im
    if mode in ("L", "LA", "P"):
        return rgb.convert(mode if mode != "P" else "RGB").convert(mode) if mode != "P" else rgb.quantize(64)
    if mode == "I;16":
        return Image.fromarray((np.asarray(rgb.convert("L"), dtype=np.uint16) * 3).astype(np.uint16))
    raise ValueError(mode)


def encode(im, fmt, rng):
    buf = io.BytesIO()
    if fmt == "JPEG":
        im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        im.save(buf, "JPEG", quality=rng.choice([60, 85, 95]), subsampling=rng.choice([0, 1, 2]))
    elif fmt == "PNG":
        im.save(buf, "PNG")
    elif fmt == "WEBP":
        im = im.convert("RGBA") if im.mode in ("LA", "P") else im
        if im.mode in ("I;16",):
            im = im.convert("RGB")
        im.save(buf, "WEBP", lossless=rng.random() < 0.4, quality=80)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--ckpt", default="/scratch/dsv4-ve")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.ckpt, "inference"))
    import image_processor as ip

    with open(os.path.join(a.ckpt, "inference", "config.json")) as f:
        cfg = json.load(f)
    args = SimpleNamespace(**{k: cfg[k] for k in cfg if k.startswith("vision_")})
    rng = random.Random(a.seed)
    cases = []
    for name in ("carrots.jpeg", "corn.jpeg"):
        with open(os.path.join(a.ckpt, "inference", "examples", "images", name), "rb") as f:
            cases.append((name, f.read()))
    sizes = [(1, 1), (13, 900), (900, 13), (2000, 200), (200, 2000), (5000, 400), (64, 64), (383, 385), (1920, 1080),
             (1080, 1920), (4032, 3024), (500, 333), (12, 3000), (3000, 12), (640, 480), (2560, 1600)]
    modes = ["RGB", "RGB", "RGBA", "L", "LA", "P", "I;16"]
    fmts = ["JPEG", "PNG", "WEBP"]
    while len(cases) < a.n:
        w, h = rng.choice(sizes) if rng.random() < 0.5 else (rng.randint(8, 3000), rng.randint(8, 3000))
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
            path = os.path.join(td, "img")
            with open(path, "wb") as f:
                f.write(data)
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"data": data}, args)
            ref = patches.contiguous().view(dtype=__import__("torch").int16).numpy().astype(np.uint16).reshape(-1)
            for start in (0, 1, 2, 3, 1001):
                out = os.path.join(td, "patches")
                r = subprocess.run([a.tool, path, str(start), out], capture_output=True, text=True)
                got = json.loads(r.stdout.strip().splitlines()[-1])
                if "error" in got:
                    print(f"{name}: ours failed: {got['error']}")
                    bad += 1
                    break
                types, perm = ip.build_image_block(n_llm_h, n_llm_w, start)
                ok_dims = (got["n_vit_h"], got["n_vit_w"], got["n_llm_h"], got["n_llm_w"]) == (n_vit_h, n_vit_w, n_llm_h, n_llm_w)
                ok_block = got["types"] == types.tolist() and got["perm"] == perm.tolist()
                ours = np.fromfile(out, dtype=np.uint16)
                same = ours.shape == ref.shape and bool((ours == ref).all())
                if start == 0 or not ok_block:
                    diff = "" if same or ours.shape != ref.shape else f", {int((ours != ref).sum())} of {ref.size} values differ"
                    print(f"{name}: grid {n_vit_h}x{n_vit_w} -> {n_llm_h}x{n_llm_w}: dims {'ok' if ok_dims else 'DIFFER'}, "
                          f"patches {'identical' if same else 'DIFFER' + diff}, block {'ok' if ok_block else 'DIFFERS'} (start {start})")
                bad += (not ok_dims) + (not same and start == 0) + (not ok_block)
    print(f"{len(cases)} images: {'all identical' if bad == 0 else f'{bad} mismatches'}")
    bad += compare_expansion(a.tool, a.ckpt, cases, rng)
    return 1 if bad else 0


def compare_expansion(tool, ckpt, cases, rng):
    """Full prompts with images: encode_messages -> tokenizer -> prepare_vl_inputs vs our expand_images."""
    import base64
    from transformers import AutoTokenizer

    sys.path.insert(0, os.path.join(ckpt, "encoding"))
    import encoding_dsv4 as E
    import image_processor as ip

    with open(os.path.join(ckpt, "inference", "config.json")) as f:
        cfg = json.load(f)
    args = SimpleNamespace(**{k: cfg[k] for k in cfg if k.startswith("vision_")}, vocab_size=cfg["vocab_size"])
    tok = AutoTokenizer.from_pretrained(ckpt)
    placeholder = tok.convert_tokens_to_ids(E.IMAGE_PLACEHOLDER)
    bad = 0
    for t in range(12):
        picks = [rng.choice(cases)[1] for _ in range(rng.randint(1, 3))]
        urls = ["data:image/png;base64," + base64.b64encode(d).decode() for d in picks]
        content = []
        for u in urls:
            if rng.random() < 0.7:
                content.append({"type": "text", "text": rng.choice(["Look:", "图片", "x" * rng.randint(1, 9)])})
            content.append({"type": "image_url", "image_url": {"url": u}})
        content.append({"type": "text", "text": "What is shown?"})
        msgs = [{"role": "user", "content": content}]
        prompt, media = E.encode_messages(msgs, thinking_mode=rng.choice(["chat", "thinking"]), return_multi_modal_data=True)
        ids = tok.encode(prompt, add_special_tokens=False)
        want_ids, want_imgs = ip.prepare_vl_inputs(prompt, media["images"], tok, args)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "in.json")
            with open(p, "w") as f:
                json.dump({"ids": ids, "urls": urls, "placeholder": placeholder, "vocab": args.vocab_size}, f)
            got = json.loads(subprocess.run([tool, "expand", p], capture_output=True, text=True).stdout)
        if "error" in got:
            print(f"expansion {t}: ours failed: {got['error']}")
            bad += 1
            continue
        same_ids = got["ids"] == list(want_ids)
        same_imgs = [(g["start"], g["n_vit_h"], g["n_vit_w"], g["perm"]) for g in got["images"]] == \
                    [(w.start, w.n_vit_h, w.n_vit_w, w.perm.tolist()) for w in want_imgs]
        print(f"expansion {t}: {len(urls)} images, {len(want_ids)} tokens: ids {'identical' if same_ids else 'DIFFER'}, "
              f"images {'identical' if same_imgs else 'DIFFER'}")
        bad += (not same_ids) + (not same_imgs)
    return bad


if __name__ == "__main__":
    sys.exit(main())

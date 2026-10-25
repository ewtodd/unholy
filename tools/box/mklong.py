"""Long distinct text for routing and prefill measurements at depth, and the character offsets where it reaches
given token depths (through the model's own tokenizer).

    $GLM_PY tools/box/mklong.py --kind pile --tokenizer /scratch/models/glm53-flash-exl3-4.05bpw/tokenizer.json \
        --out /work/data/glm/long-pile.txt

Kinds: pile (the Pile validation documents in order, from the Hugging Face cache; web, books, papers, code, mail),
pycode (the Python standard library's .py files in sorted path order, each under a "# file:" line), repeat (--src's
text over and over, the way a curve over a short id file repeats). The text stops at the last depth; OUT.cuts holds
"tokens chars" per depth (the content cut there has about that many tokens; the chat template adds a few).
"""
import argparse
import glob
import os

from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--kind", required=True, choices=["pile", "pycode", "repeat"])
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--src", help="repeat: the text file repeated")
ap.add_argument("--depths", default="16384,32768,65536,131072,262144,524288")
ap.add_argument("--skip", type=int, default=0, help="documents / files left out first (held-out text)")
args = ap.parse_args()
depths = [int(d) for d in args.depths.split(",")]
want_chars = depths[-1] * 6  # text gathered: GLM averages 3-4.5 characters per token


def pieces():
    if args.kind == "pile":
        import datasets

        ds = datasets.load_dataset("mit-han-lab/pile-val-backup", split="validation")
        for row in ds:
            yield row["text"]
    elif args.kind == "pycode":
        lib = sorted(glob.glob("/nix/store/*-python3-3.1[0-9].*/lib/python3.1[0-9]"))[0]
        for p in sorted(glob.glob(os.path.join(lib, "**", "*.py"), recursive=True)):
            if "/site-packages/" in p or "/test" in p:
                continue
            with open(p, encoding="utf-8", errors="replace") as f:
                yield f"# file: {os.path.relpath(p, lib)}\n{f.read()}"
    else:
        with open(args.src, encoding="utf-8") as f:
            src = f.read()
        while True:
            yield src


parts, n = [], 0
for i, p in enumerate(pieces()):
    if args.kind != "repeat" and i < args.skip:
        continue
    parts.append(p)
    n += len(p) + 2
    if n >= want_chars:
        break
text = "\n\n".join(parts)
tok = Tokenizer.from_file(args.tokenizer)
enc = tok.encode(text, add_special_tokens=False)
if len(enc.ids) < depths[-1]:
    raise SystemExit(f"only {len(enc.ids)} tokens of text, need {depths[-1]}")
cuts = [enc.offsets[d][0] for d in depths]
with open(args.out, "w", encoding="utf-8") as f:
    f.write(text[: cuts[-1]])
with open(args.out + ".cuts", "w") as f:
    for d, c in zip(depths, cuts):
        f.write(f"{d} {c}\n")
print(f"{args.out}: {cuts[-1]} chars, {depths[-1]} tokens ({cuts[-1] / depths[-1]:.2f} chars/token)")

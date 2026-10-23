"""Token ids for the oracle, the benches and the curves: code text through the model's own tokenizer.

    $REF_PY tools/box/mkids.py --tokenizer /scratch/dsv4-ve/tokenizer.json --src /work/unholy-perf \
        --corpus /work/data/code-corpus.txt --out /work/data/ids_code32768.bin

The corpus is the tree's C++/HIP sources in sorted path order. It is written next to the ids and reused
when it exists, so the ids can be regenerated exactly after the tree has changed. Output: uint32, BOS first.
"""
import argparse
import os

import numpy as np
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--src", required=True, help="tree whose src/ and tools/ sources make the corpus")
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--count", type=int, default=32768)
ap.add_argument("--bos", type=int, default=0)
args = ap.parse_args()

if not os.path.exists(args.corpus):
    paths = []
    for top in ("src", "tools"):
        for root, _, files in os.walk(os.path.join(args.src, top)):
            paths += [os.path.join(root, f) for f in files if f.endswith((".cpp", ".hpp", ".hip"))]
    parts = []
    for p in sorted(paths):
        with open(p, encoding="utf-8") as f:
            parts.append(f"// {os.path.relpath(p, args.src)}\n{f.read()}")
    with open(args.corpus, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

with open(args.corpus, encoding="utf-8") as f:
    text = f.read()
ids = [args.bos] + Tokenizer.from_file(args.tokenizer).encode(text, add_special_tokens=False).ids
if len(ids) < args.count:
    raise SystemExit(f"corpus has only {len(ids)} tokens, need {args.count}")
np.array(ids[: args.count], dtype=np.uint32).tofile(args.out)
print(f"{len(ids)} tokens in {args.corpus}; wrote the first {args.count} to {args.out}")

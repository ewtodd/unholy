"""A routing profile mixed from several (usage-profile format, "L<layer> c0 c1 ..."): each source normalized per
layer, then weighted and summed, so every source counts the same per layer however many tokens it saw.

    python3 tools/box/mixprofile.py --out data/expert_profile_glm53_broad.txt \
        --sum RUN_PILE/profile_16384.txt,RUN_PILE/profile_32768.txt,...:1 OLD_PROFILE.txt:0.5

A positional FILE[:WEIGHT] is one source (weight 1 by default); --sum FILE,FILE,...[:WEIGHT] sums several files
(a run's depth segments) into one source. Layers no source routes (dense ones) come out as zeros.
"""
import argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--sum", action="append", default=[], help="FILE[,FILE...][:WEIGHT]: files summed into one source")
ap.add_argument("inputs", nargs="*")
args = ap.parse_args()


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) > 1 and p[0].startswith("L"):
                rows[p[0]] = [float(v) for v in p[1:]]
    return rows


def split_weight(spec):
    if ":" in spec and spec.rsplit(":", 1)[1].replace(".", "", 1).isdigit():
        path, w = spec.rsplit(":", 1)
        return path, float(w)
    return spec, 1.0


sources = []
for spec in args.inputs:
    path, w = split_weight(spec)
    sources.append((load(path), w))
for spec in args.sum:
    paths, w = split_weight(spec)
    acc = defaultdict(list)
    for path in paths.split(","):
        for k, v in load(path).items():
            acc[k] = [a + b for a, b in zip(acc[k], v)] if acc[k] else v
    sources.append((dict(acc), w))

layers = sorted({k for rows, _ in sources for k in rows}, key=lambda k: int(k[1:]))
with open(args.out, "w") as f:
    for k in layers:
        mix = None
        for rows, w in sources:
            if k not in rows or sum(rows[k]) == 0:
                continue
            s = sum(rows[k])
            v = [w * c / s for c in rows[k]]
            mix = v if mix is None else [a + b for a, b in zip(mix, v)]
        if mix is None:
            mix = [0.0] * len(next(rows[k] for rows, _ in sources if k in rows))
        # integer counts on a 1e6 scale per layer (the planner normalizes per layer)
        f.write(k + " " + " ".join(str(int(round(c * 1e6))) for c in mix) + "\n")
print(f"{args.out}: {len(layers)} layers from {len(sources)} sources")

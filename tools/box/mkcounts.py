# Per-expert row counts for the APU-kernel benchmarks (unholy-mx-test --counts, gufo-routed-bench --counts):
# one line of integers, rows per expert. From a routing profile and the planner's greedy hot set (as plan.cpp).
#   mkcounts.py PROFILE TOPK HOT0 HOT1 SPLIT PART_ROWS LAYER SEED > file
# HOT0 / HOT1: the plan's hot experts per card, SPLIT: card 1's first layer. The box's data/apu0 files:
#   mkcounts.py data/expert_profile_unholy_mixed.txt 6 1541 1454 22 2048 20 1 > dsv4-apu-l20-part2048.txt (4096: part4096)
import sys, random, math
prof, topk, hot0, hot1, split, part, layer, seed = sys.argv[1], *map(int, sys.argv[2:])
layers = []
for line in open(prof):
    if line.startswith("L"):
        v = [float(x) for x in line.split()[1:]]; s = sum(v)
        layers.append([x / s for x in v] if s else [0.0] * len(v))
cands = [[], []]
for l, sh in enumerate(layers):
    for e, x in enumerate(sh): cands[0 if l < split else 1].append((x, l, e))
hot = set()
for k, n in ((0, hot0), (1, hot1)):
    for x, l, e in sorted(cands[k], reverse=True)[:n]: hot.add((l, e))
rng = random.Random(seed)
def poisson(lam):
    if lam > 50: return max(0, int(rng.gauss(lam, math.sqrt(lam)) + 0.5))
    x, p = 0, math.exp(-lam); s = p; u = rng.random()
    while u > s: x += 1; p *= lam / x; s += p
    return x
rows = [poisson(part * topk * x) for e, x in enumerate(layers[layer]) if (layer, e) not in hot]
print(" ".join(str(r) for r in rows if r > 0))

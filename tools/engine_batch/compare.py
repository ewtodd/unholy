#!/usr/bin/env python3
"""Compare unholy-engine-batch outputs (run with --logprobs 1 for fingerprints).

  compare.py A.jsonl B.jsonl
      tokens and logprob fingerprints of two runs, request by request (first differing token).
  compare.py --modal "A1.jsonl A2.jsonl ..." "B1.jsonl B2.jsonl ..."
      each build's most common fingerprint per request over its runs, its outlier runs, and whether the two
      builds' majority fingerprints agree.
"""
import collections, json, sys


def load(path):
    return {r["index"]: (tuple(r["tokens"]), r.get("logprobs", "")) for r in map(json.loads, open(path))}


def pair(a_path, b_path):
    a, b = load(a_path), load(b_path)
    tok = lp = 0
    diffs = []
    for i in sorted(a):
        (ta, la), (tb, lb) = a[i], b[i]
        tok += ta == tb
        lp += la == lb
        if la != lb or ta != tb:
            n = min(len(ta), len(tb))
            k = next((j for j in range(n) if ta[j] != tb[j] or la[8 * j:8 * j + 8] != lb[8 * j:8 * j + 8]), n)
            diffs.append(f"{i}@{k}")
    print(f"tokens {tok}/{len(a)}, logprobs {lp}/{len(a)}" + (f"; first difference (request@token): {' '.join(diffs)}" if diffs else ""))


def modal(paths):
    runs = [load(p) for p in paths]
    mode, outliers = {}, []
    for i in sorted(runs[0]):
        c = collections.Counter(r[i] for r in runs)
        mode[i], n = c.most_common(1)[0]
        outliers += [f"{p}:request {i}" for p, r in zip(paths, runs) if r[i] != mode[i]]
        if 2 * n <= len(runs):
            outliers.append(f"request {i}: no majority")
    return mode, outliers


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--modal":
        A, B = sys.argv[2].split(), sys.argv[3].split()
        ma, oa = modal(A)
        mb, ob = modal(B)
        print(f"A: {len(A)} runs, outliers: {oa or 'none'}")
        print(f"B: {len(B)} runs, outliers: {ob or 'none'}")
        print(f"majority fingerprints identical for {sum(ma[i] == mb[i] for i in ma)} / {len(ma)} requests")
    elif len(sys.argv) == 3:
        pair(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)

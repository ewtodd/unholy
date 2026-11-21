"""Idle time between kernels in a rocprofv3 kernel trace, per device, bucketed by gap length: short gaps are
launch latency between back-to-back kernels, long ones are waits (the host between steps, the other card, the
APU). Also the largest gaps with the kernels on either side.

    python3 tools/box/trace-gaps.py RUN_kernel_trace.csv [--last SECONDS] [--steps N] [--top 12]
"""
import argparse
import csv
import re
from collections import defaultdict


def short(name):
    name = name.replace("(anonymous namespace)::", "")
    name = re.sub(r"^void ", "", name)
    name = re.sub(r"unholy::(kernels::|model::|engine::)?", "", name)
    return re.sub(r"\(.*", "", name)[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--last", type=float)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    rows = defaultdict(list)
    with open(a.trace) as f:
        for r in csv.DictReader(f):
            rows[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), short(r["Kernel_Name"])))
    last = max(e for v in rows.values() for _, e, _ in v)
    lo = last - int(a.last * 1e9) if a.last else 0
    edges = [0, 5e3, 10e3, 20e3, 50e3, 100e3, 500e3, 2e6, float("inf")]  # ns
    for agent in sorted(rows):
        ks = sorted(k for k in rows[agent] if k[0] >= lo)
        if len(ks) < 2:
            continue
        buckets = [[0, 0.0] for _ in range(len(edges) - 1)]
        big = []
        busy_end = ks[0][1]
        for prev, cur in zip(ks, ks[1:]):
            busy_end = max(busy_end, prev[1])
            gap = cur[0] - busy_end
            if gap <= 0:
                continue
            for i in range(len(edges) - 1):
                if edges[i] <= gap < edges[i + 1]:
                    buckets[i][0] += 1
                    buckets[i][1] += gap
                    break
            big.append((gap, prev[2], cur[2]))
        n = a.steps
        print(f"Agent {agent}: gaps per step by length")
        for i, (cnt, tot) in enumerate(buckets):
            hi = "inf" if edges[i + 1] == float("inf") else f"{edges[i + 1] / 1e3:g}"
            print(f"  {edges[i] / 1e3:>6g} .. {hi:>5s} us: {cnt / n:8.1f} gaps  {tot / n / 1e6:7.3f} ms")
        big.sort(reverse=True)
        print("  largest:")
        for gap, p, c in big[: a.top]:
            print(f"    {gap / 1e3:9.1f} us  after {p}  before {c}")


if __name__ == "__main__":
    main()

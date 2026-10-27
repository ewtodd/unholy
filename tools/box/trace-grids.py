#!/usr/bin/env python3
"""Per-kernel grids of a rocprofv3 kernel trace (csv): launches, blocks, block size, VGPRs, LDS and average time over
the trace's last seconds, flagging grids that fill a card's wave slots exactly (on gfx1201 such a short kernel takes
~22 us to complete however little it does; DECISIONS 2026-10-27).

  tools/box/trace-grids.py RUN/run_kernel_trace.csv [--last 2.5] [--steps 60] [--top 40] [--pattern NAME] [--cus 64]
"""
import argparse
import collections
import csv

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--last", type=float, default=2.5, help="seconds at the end of the trace")
ap.add_argument("--steps", type=int, default=60, help="steps in that window (per-step figures)")
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--pattern", default="", help="only kernels whose name contains this")
ap.add_argument("--cus", type=int, default=64, help="CUs of the discrete cards")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.trace)))
t1 = max(int(r["End_Timestamp"]) for r in rows)
t0 = t1 - a.last * 1e9
agg = collections.defaultdict(lambda: [0, 0.0])
for r in rows:
    if int(r["Start_Timestamp"]) < t0 or a.pattern not in r["Kernel_Name"]:
        continue
    dims = [(int(r["Grid_Size_" + d]), max(1, int(r["Workgroup_Size_" + d]))) for d in "XYZ"]
    wg = dims[0][1] * dims[1][1] * dims[2][1]
    if wg <= 1:  # runtime stream operations
        continue
    blocks = 1
    for g, w in dims:
        blocks *= max(1, g // w)
    name = r["Kernel_Name"].replace("(anonymous namespace)", "").split("(")[0].split("::")[-1]
    k = (r["Agent_Id"], name[:48], blocks, wg, int(r["VGPR_Count"]), int(r["LDS_Block_Size"]))
    agg[k][0] += 1
    agg[k][1] += (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3

print("agent    kernel                                            blocks   wg vgpr    lds  per step   avg us")
for (agent, name, blocks, wg, vgpr, lds), (n, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
    waves = (wg + 31) // 32
    by_waves = 2 * min(16, 1536 // max(vgpr, 8)) // waves  # blocks per CU: 16 waves per SIMD at most
    by_lds = 65536 // lds if lds else by_waves
    cap = a.cus * min(by_waves, by_lds)
    full = agent != "Agent 3" and by_waves <= by_lds and cap > 0 and any(
        m * cap - 3 <= blocks <= m * cap for m in (1, 2))
    print("%-8s %-48s %6d %4d %4d %6d %9.1f %8.1f%s" % (agent, name, blocks, wg, vgpr, lds, n / a.steps, us / n,
                                                         "  FULL GRID" if full else ""))

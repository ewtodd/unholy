"""Where the GPU time goes in a rocprofv3 kernel trace: per device, how busy it was over a window, and its kernels by
total time.

    python3 tools/box/trace-summary.py RUN_kernel_trace.csv [--last SECONDS | --from S --to S] [--top 25] [--steps N]

--last takes the final SECONDS of the trace (e.g. a run's decode steps after its prefill); --from / --to are seconds
from the trace's first kernel. --steps divides the totals by N (per-step figures). Busy is the union of the device's
kernel intervals over the window.
"""
import argparse
import csv
import re
from collections import defaultdict


def short(name):
    name = name.replace("(anonymous namespace)::", "")
    name = re.sub(r"^void ", "", name)
    name = re.sub(r"unholy::(kernels::|model::|engine::)?", "", name)
    name = re.sub(r"\(.*", "", name)
    return name[:90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--last", type=float)
    ap.add_argument("--from", dest="t0", type=float)
    ap.add_argument("--to", dest="t1", type=float)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--steps", type=int, default=1)
    a = ap.parse_args()
    rows = []
    with open(a.trace) as f:
        for r in csv.DictReader(f):
            rows.append((r["Agent_Id"], short(r["Kernel_Name"]), int(r["Start_Timestamp"]), int(r["End_Timestamp"])))
    first = min(r[2] for r in rows)
    last = max(r[3] for r in rows)
    if a.last is not None:
        lo, hi = last - int(a.last * 1e9), last
    else:
        lo = first + int((a.t0 or 0) * 1e9)
        hi = first + int(a.t1 * 1e9) if a.t1 is not None else last
    win = [r for r in rows if r[2] >= lo and r[3] <= hi]
    span = (hi - lo) / 1e9
    print(f"window {span:.3f} s ({(lo - first) / 1e9:.3f} .. {(hi - first) / 1e9:.3f} s), {len(win)} kernels, "
          f"per-step figures over {a.steps} steps")
    by_agent = defaultdict(list)
    for r in win:
        by_agent[r[0]].append(r)
    for agent in sorted(by_agent):
        ks = sorted(by_agent[agent], key=lambda r: r[2])
        busy, cur_s, cur_e = 0, None, None
        for _, _, s, e in ks:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            busy += cur_e - cur_s
        tot = defaultdict(lambda: [0, 0])
        for _, n, s, e in ks:
            tot[n][0] += e - s
            tot[n][1] += 1
        print(f"\n{agent}: busy {busy / 1e9 / span:.1%} ({busy / 1e6 / a.steps:.2f} ms per step), "
              f"{len(ks) / a.steps:.0f} kernels per step, kernel time {sum(v[0] for v in tot.values()) / 1e6 / a.steps:.2f} ms per step")
        print(f"  {'ms/step':>8} {'count/step':>10} {'avg us':>8}  kernel")
        for n, (t, c) in sorted(tot.items(), key=lambda kv: -kv[1][0])[: a.top]:
            print(f"  {t / 1e6 / a.steps:8.3f} {c / a.steps:10.1f} {t / c / 1e3:8.1f}  {n}")


if __name__ == "__main__":
    main()

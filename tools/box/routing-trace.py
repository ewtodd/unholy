#!/usr/bin/env python3
"""Reader and summary of an UNHOLY_ROUTING_TRACE file (DSV4 resident path).

File: "UHROUTE1", u16 n_layers, n_mtp, n_active, n_experts; then records
  u32 magic 0x31525455, u8 kind (0 target step, 1 drafter block), u8 pad, u16 layers, u16 layer0, u16 n_active,
  u32 pad, u64 step, u32 rows, u32 cut (a two-part step's first row of part 2; 0: whole),
  rows x {u64 seq, u32 pos, u16 slot, u8 row_kind, u8 pad}   row_kind: 0 prefill, 1 decode, 2 verify, 3 drafter
  uint8 ids[layers][rows][n_active]
A target step has all n_layers layers (layer0 0); a drafter record the n_mtp layers 43..45 (seq 0: unknown).

  routing-trace.py summary TRACE [--profile-out FILE] [--compare PROFILE]

As a library (tools/sim/cache_sim.py loads it by path): open_trace(path) -> Trace.
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = 0x31525455
HEADER = struct.Struct("<IBBHHHIQII")  # 32 bytes
ROW = np.dtype([("seq", "<u8"), ("pos", "<u4"), ("slot", "<u2"), ("kind", "u1"), ("pad", "u1")])
ROW_KINDS = ["prefill", "decode", "verify", "drafter"]
assert HEADER.size == 32 and ROW.itemsize == 16


class Record:
    __slots__ = ("trace", "off", "kind", "layers", "layer0", "step", "rows", "cut")

    def __init__(self, trace, off, kind, layers, layer0, step, rows, cut):
        self.trace, self.off, self.kind, self.layers, self.layer0 = trace, off, kind, layers, layer0
        self.step, self.rows, self.cut = step, rows, cut

    @property
    def meta(self):
        """Structured array [rows] (seq, pos, slot, kind)."""
        a = self.off + HEADER.size
        return np.frombuffer(self.trace.buf, dtype=ROW, count=self.rows, offset=a)

    @property
    def ids(self):
        """uint8 [layers, rows, k]: expert ids."""
        a = self.off + HEADER.size + self.rows * ROW.itemsize
        k = self.trace.k
        return np.frombuffer(self.trace.buf, dtype=np.uint8, count=self.layers * self.rows * k, offset=a).reshape(
            self.layers, self.rows, k)


class Trace:
    def __init__(self, path):
        self.path = path
        self.buf = np.memmap(path, dtype=np.uint8, mode="r")
        if bytes(self.buf[:8]) != b"UHROUTE1":
            raise ValueError(f"{path}: not a routing trace")
        self.n_layers, self.n_mtp, self.k, self.n_experts = struct.unpack_from("<4H", self.buf, 8)
        self.records = []
        off, size = 16, len(self.buf)
        while off + HEADER.size <= size:
            magic, kind, _, layers, layer0, k, _, step, rows, cut = HEADER.unpack_from(self.buf, off)
            if magic != MAGIC or k != self.k:
                raise ValueError(f"{path}: bad record at {off}")
            n = HEADER.size + rows * ROW.itemsize + layers * rows * k
            if off + n > size:  # truncated tail (the server was killed): ignore it
                break
            self.records.append(Record(self, off, kind, layers, layer0, step, rows, cut))
            off += n
        self.truncated = off < size

    @property
    def n_total_layers(self):
        return self.n_layers + self.n_mtp

    def histogram(self, drafter=False):
        """int64 [layers, experts]: picks per expert over all target steps (drafter: the drafter records)."""
        nl = self.n_mtp if drafter else self.n_layers
        h = np.zeros((nl, self.n_experts), np.int64)
        for r in self.records:
            if (r.kind == 1) != drafter:
                continue
            ids = r.ids
            for l in range(r.layers):
                h[l] += np.bincount(ids[l].ravel(), minlength=self.n_experts)
        return h


def open_trace(path):
    return Trace(path)


def read_profile(path):
    """The --profile / routing_profile text format: 'L<i> c0 c1 ...' per layer -> int64 [layers, experts]."""
    rows = []
    for line in open(path):
        p = line.split()
        if p and p[0].startswith("L"):
            rows.append([int(x) for x in p[1:]])
    return np.array(rows, np.int64)


def write_profile(h, path):
    with open(path, "w") as f:
        for l in range(len(h)):
            f.write(f"L{l} " + " ".join(str(int(x)) for x in h[l]) + "\n")


def summary(t, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p(f"{t.path}: {t.n_layers} layers + {t.n_mtp} drafter, k={t.k}, {t.n_experts} experts, {len(t.records)} records"
      + (" (truncated tail ignored)" if t.truncated else ""))
    kinds = np.zeros(4, np.int64)
    steps = [0, 0]
    seqs = {}
    stepsize = []
    for r in t.records:
        steps[r.kind] += 1
        m = r.meta
        kinds += np.bincount(m["kind"], minlength=4)
        if r.kind == 0:
            stepsize.append(r.rows)
            for s in np.unique(m["seq"]):
                seqs.setdefault(int(s), 0)
                seqs[int(s)] += int((m["seq"] == s).sum())
    p(f"target steps {steps[0]}, drafter blocks {steps[1]}; rows: " +
      ", ".join(f"{n} {int(c)}" for n, c in zip(ROW_KINDS, kinds)) + f"; sequences {len(seqs)}")
    if stepsize:
        ss = np.array(stepsize)
        p(f"step rows: mean {ss.mean():.1f}, median {int(np.median(ss))}, max {ss.max()}")
    h = t.histogram()
    tot = h.sum(1)
    p(f"picks per layer: min {tot.min()}, max {tot.max()} (= rows x {t.k} in a consistent trace)")
    used = (h > 0).sum(1)
    p(f"experts ever used per layer: mean {used.mean():.1f}, min {used.min()}")
    srt = -np.sort(-h, axis=1)
    cs = np.cumsum(srt, axis=1) / np.maximum(tot[:, None], 1)
    for frac in (0.1, 0.3, 0.6):
        n = int(round(frac * t.n_experts))
        p(f"picks caught by the top {int(frac * 100)}% of experts of this trace itself: {cs[:, n - 1].mean() * 100:.1f}%")
    if t.n_mtp and steps[1]:
        hd = t.histogram(drafter=True)
        p(f"drafter picks per layer: {hd.sum(1).tolist()}")
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("trace")
    s.add_argument("--profile-out", help="write the target layers' histogram in the profile format")
    s.add_argument("--compare", help="a /unholy/routing_profile dump of the same server run: counts must be equal")
    a = ap.parse_args()
    t = open_trace(a.trace)
    h = summary(t)
    if a.profile_out:
        write_profile(h, a.profile_out)
    if a.compare:
        ref = read_profile(a.compare)
        if ref.shape != h.shape:
            print(f"compare: shape {ref.shape} vs trace {h.shape}")
            sys.exit(1)
        bad = (ref != h)
        if bad.any():
            l, e = np.argwhere(bad)[0]
            print(f"compare: MISMATCH in {int(bad.sum())} cells, first layer {l} expert {e}: profile {ref[l, e]} trace {h[l, e]}")
            sys.exit(1)
        print(f"compare: identical ({int(h.sum())} picks, {h.shape[0]} layers)")


if __name__ == "__main__":
    main()

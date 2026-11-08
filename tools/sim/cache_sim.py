#!/usr/bin/env python3
"""Replay UNHOLY_ROUTING_TRACE files through expert-cache policies (docs/PLAN-strix-only.md, P0.1).

  cache_sim.py TRACE... [--profile data/expert_profile_unholy_mixed.txt] [--sizes 80 95 110]
               [--policies static lru gufo hotness] [--rechunk 1024 2048 4096 8192] [--warm TRACE] [--json OUT]

A workload is one trace file, replayed from a cold start (static and gufo start from the profile) as one session.
A cache holds whole experts (13.37 MB each, GB = 1e9 bytes) over all layers, drafter layers included when the
trace has them. Per batch (one step, one layer) a "read" is one distinct expert the batch uses; a hit is a read the
cache holds. Policies:
  static    top-N experts per layer from the profile (drafter layers: from the trace itself, no profile has them)
  lru       global LRU, admit every miss
  gufo      score/(1 + age/1600) with age in prepares (batch-layer calls), each use adds min(rows, 4), +4 for the
            batch's last 16 rows; seeded from the profile (15/16 of the budget, min(8, log2(1 + count)));
            when full a miss is admitted only if the batch has <= 16 rows, or the expert is in the last 16 rows, or
            rows >= max(4, n/32), else it goes through a transient stage; never evicts what the batch uses
  hotness   ds4-Metal style: selection counts (rows) halved every 16 decode tokens, evict lowest, LRU on ties
Step classes: a step with any prefill row is a prefill step (its misses are the bytes read); the others are decode
steps (decode and verify rows); drafter blocks count with the decode steps.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

EXPERT_MB = 13.37
EXPERT_BYTES = EXPERT_MB * 1e6

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("routing_trace", os.path.join(_here, "..", "box", "routing-trace.py"))
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


# ---------------------------------------------------------------------------------------------------------------
# Batches

class Batch:
    """One model step (or drafter block): ids uint8 [layers, rows, k] for layers layer0.., row kinds, sequences."""
    __slots__ = ("kind", "layer0", "ids", "rkind", "seq", "pos", "committed")

    def __init__(self, kind, layer0, ids, rkind, seq, pos, committed=0):
        self.kind, self.layer0, self.ids, self.rkind, self.seq, self.pos = kind, layer0, ids, rkind, seq, pos
        self.committed = committed  # decode tokens this step moved forward (decode steps)

    @property
    def rows(self):
        return self.ids.shape[1]


def _cls(rkind):
    return "prefill" if (rkind == 0).any() else "decode"


def batches_of(trace, rechunk=0):
    """The trace as batches in file order. rechunk > 0: the prefill rows of each sequence regrouped into steps of
    that many rows (routing is per token, so this is exact); decode steps are kept."""
    recs = trace.records
    # tokens committed per decode / verify chunk: the next chunk's start of the sequence, or 1
    chunks = []  # (record index, seq, start, n)
    for i, r in enumerate(recs):
        if r.kind != 0:
            continue
        m = r.meta
        if not (m["kind"] != 0).any():
            continue
        j = 0
        while j < r.rows:
            s, k = int(m["seq"][j]), int(m["kind"][j])
            e = j + 1
            while e < r.rows and int(m["seq"][e]) == s and int(m["kind"][e]) == k:
                e += 1
            if k != 0:
                chunks.append((i, s, int(m["pos"][j]), e - j))
            j = e
    committed = {}
    last = {}
    for ci in range(len(chunks) - 1, -1, -1):
        i, s, start, n = chunks[ci]
        nxt = last.get(s)
        c = min(max(nxt - start, 1), n) if nxt is not None and nxt > start else 1
        committed[i] = committed.get(i, 0) + c
        last[s] = start
    pending = {}  # seq -> list of (ids [L, n, k], pos [n])

    def flush():
        for s in list(pending):
            parts = pending.pop(s)
            ids = np.concatenate([p[0] for p in parts], axis=1)
            pos = np.concatenate([p[1] for p in parts])
            for a in range(0, ids.shape[1], rechunk):
                b = min(a + rechunk, ids.shape[1])
                yield Batch("prefill", 0, ids[:, a:b], np.zeros(b - a, np.uint8), np.full(b - a, s, np.uint64), pos[a:b])

    for i, r in enumerate(recs):
        m = r.meta
        ids = r.ids
        if r.kind == 1:
            yield from flush() if rechunk else ()
            yield Batch("decode", r.layer0, ids, m["kind"], m["seq"], m["pos"], 0)
            continue
        if not rechunk:
            yield Batch(_cls(m["kind"]), 0, ids, m["kind"], m["seq"], m["pos"], committed.get(i, 0))
            continue
        pre = m["kind"] == 0
        if pre.any():
            for s in dict.fromkeys(int(x) for x in m["seq"][pre]):
                sel = pre & (m["seq"] == s)
                pending.setdefault(s, []).append((ids[:, sel], np.array(m["pos"][sel])))
        if (~pre).any():
            yield from flush()
            yield Batch("decode", 0, ids[:, ~pre], m["kind"][~pre], m["seq"][~pre], m["pos"][~pre], committed.get(i, 0))
    if rechunk:
        yield from flush()


def layer_events(b, n_experts):
    """(layer, experts, rows using each, in the last 16 rows) for each layer of the batch."""
    last = b.ids[:, -16:, :]
    for l in range(b.ids.shape[0]):
        cnt = np.bincount(b.ids[l].ravel(), minlength=n_experts)
        ex = np.flatnonzero(cnt)
        l16 = np.bincount(last[l].ravel(), minlength=n_experts)[ex] > 0
        yield b.layer0 + l, ex, cnt[ex], l16


# ---------------------------------------------------------------------------------------------------------------
# Policies. access() returns the hit mask over `experts` and updates the cache.

class Policy:
    def __init__(self, capacity, n_layers, n_experts):
        self.cap, self.E, self.nkeys = int(capacity), n_experts, n_layers * n_experts
        self.inc = np.zeros(self.nkeys, bool)
        self.n = 0
        self.now = 0

    def _victims(self, m, keys, rank):
        """m cached keys not in `keys` with the lowest `rank` (an array over all keys)."""
        cand = np.flatnonzero(self.inc)
        cand = cand[~np.isin(cand, keys)]
        if m >= len(cand):
            return cand
        return cand[np.argpartition(rank[cand], m - 1)[:m]]

    def _insert(self, miss, keys, rank):
        """Admit `miss` keys (already filtered), evicting the lowest-rank cached keys outside the batch when
        full; returns the keys that fit."""
        if len(miss) == 0:
            return miss
        free = self.cap - self.n
        need = len(miss) - free
        if need > 0:
            v = self._victims(need, keys, rank)
            self.inc[v] = False
            self.n -= len(v)
            miss = miss[: self.cap - self.n]
        self.inc[miss] = True
        self.n += len(miss)
        return miss

    def note_decode(self, tokens):
        pass


class Static(Policy):
    def __init__(self, capacity, n_layers, n_experts, counts):
        super().__init__(capacity, n_layers, n_experts)
        used = int((counts.sum(1) > 0).sum()) or 1
        base, extra = divmod(self.cap, used)
        li = 0
        for l in range(n_layers):
            if counts[l].sum() == 0:
                continue
            n = base + (1 if li < extra else 0)
            li += 1
            top = np.argsort(-counts[l], kind="stable")[:n]
            self.inc[l * n_experts + top] = True
        self.n = int(self.inc.sum())

    def access(self, layer, ex, cnt, l16, rows):
        self.now += 1
        return self.inc[layer * self.E + ex]


class LRU(Policy):
    def __init__(self, capacity, n_layers, n_experts):
        super().__init__(capacity, n_layers, n_experts)
        self.stamp = np.zeros(self.nkeys, np.int64)

    def access(self, layer, ex, cnt, l16, rows):
        self.now += 1
        keys = layer * self.E + ex
        hit = self.inc[keys]
        self.stamp[keys[hit]] = self.now
        miss = keys[~hit]
        miss = self._insert(miss, keys, self.stamp)
        self.stamp[miss] = self.now
        return hit


class Gufo(Policy):
    DECAY = 1600.0

    def __init__(self, capacity, n_layers, n_experts, counts):
        super().__init__(capacity, n_layers, n_experts)
        self.score = np.zeros(self.nkeys)
        self.stamp = np.zeros(self.nkeys, np.int64)
        flat = np.zeros(self.nkeys, np.int64)
        flat[: counts.size] = counts.ravel()
        n = min(self.cap * 15 // 16, int((flat > 0).sum()))
        top = np.argsort(-flat, kind="stable")[:n]
        self.inc[top] = True
        self.n = n
        self.score[top] = np.minimum(8.0, np.log2(1.0 + flat[top]))

    def eff(self, keys):
        return self.score[keys] / (1.0 + (self.now - self.stamp[keys]) / self.DECAY)

    def access(self, layer, ex, cnt, l16, rows):
        self.now += 1
        keys = layer * self.E + ex
        add = np.minimum(cnt, 4) + 4.0 * l16
        hit = self.inc[keys]
        hk = keys[hit]
        self.score[hk] = self.eff(hk) + add[hit]
        self.stamp[hk] = self.now
        mi = np.flatnonzero(~hit)
        if len(mi):
            mi = mi[np.argsort(-cnt[mi], kind="stable")]
            free = self.cap - self.n
            ok = np.ones(len(mi), bool)
            ok[free:] = (rows <= 16) | l16[mi[free:]] | (cnt[mi[free:]] >= max(4, rows / 32))
            mk = keys[mi[ok]]
            ma = add[mi[ok]]
            cand = np.flatnonzero(self.inc)
            cand = cand[~np.isin(cand, keys)]
            fit = min(len(mk), free + len(cand))
            mk, ma = mk[:fit], ma[:fit]
            need = len(mk) - free
            if need > 0:
                v = cand[np.argpartition(self.eff(cand), need - 1)[:need]] if need < len(cand) else cand
                self.inc[v] = False
                self.n -= len(v)
            self.inc[mk] = True
            self.n += len(mk)
            self.score[mk] = ma
            self.stamp[mk] = self.now
        return hit


class Hotness(Policy):
    def __init__(self, capacity, n_layers, n_experts):
        super().__init__(capacity, n_layers, n_experts)
        self.hot = np.zeros(self.nkeys, np.int64)
        self.stamp = np.zeros(self.nkeys, np.int64)
        self.dec = 0

    def access(self, layer, ex, cnt, l16, rows):
        self.now += 1
        keys = layer * self.E + ex
        self.hot[keys] += cnt
        hit = self.inc[keys]
        self.stamp[keys[hit]] = self.now
        rank = (self.hot << 32) + self.stamp
        miss = self._insert(keys[~hit], keys, rank)
        self.stamp[miss] = self.now
        return hit

    def note_decode(self, tokens):
        before = self.dec // 16
        self.dec += tokens
        for _ in range(self.dec // 16 - before):
            self.hot >>= 1


# ---------------------------------------------------------------------------------------------------------------
# Replay

class Stats:
    def __init__(self, n_layers):
        self.dec = []   # (rows, committed, reads, hits, cum committed) per decode step
        self.pre = []   # (rows, reads, hits) per prefill step
        self.dec_layer = np.zeros((n_layers, 2), np.int64)  # reads, hits
        self.draft = np.zeros(2, np.int64)
        self.verify_rows = 0
        self.decode_rows = 0
        self.verify_steps = []  # (rows of verify chunks, misses) per step with verify rows


def run(trace, policy, rechunk=0, count=True, stats=None):
    stats = stats or Stats(trace.n_total_layers)
    cum = 0
    for b in batches_of(trace, rechunk):
        reads = hits = 0
        for layer, ex, cnt, l16, in layer_events(b, trace.n_experts):
            hit = policy.access(layer, ex, cnt, l16, b.rows)
            reads += len(ex)
            h = int(hit.sum())
            hits += h
            if count and b.kind == "decode":
                stats.dec_layer[layer] += (len(ex), h)
        if not count:
            if b.kind == "decode":
                policy.note_decode(int((b.rkind != 0).sum()))
            continue
        if b.kind == "prefill":
            stats.pre.append((b.rows, reads, hits))
        else:
            if b.layer0 > 0:  # drafter block
                stats.draft += (reads, hits)
            else:
                cum += b.committed
                nv = int((b.rkind == 2).sum())
                stats.decode_rows += int((b.rkind == 1).sum())
                stats.verify_rows += nv
                if nv:
                    stats.verify_steps.append((nv, reads - hits))
            stats.dec.append((b.rows, b.committed, reads, hits, cum, 1 if b.layer0 == 0 else 0))
        if b.kind == "decode":
            policy.note_decode(int((b.rkind != 0).sum()))
    return stats


WARM_EDGES = [32, 128, 512, 2048, 8192, 32768]


def report(stats, name, n_layers):
    d = np.array(stats.dec, np.int64).reshape(-1, 6)
    p = np.array(stats.pre, np.int64).reshape(-1, 3)
    out = {}
    reads, hits = int(d[:, 2].sum()), int(d[:, 3].sum())
    out["decode_reads"], out["decode_hit_rate"] = reads, (hits / reads if reads else None)
    tokens = int(d[:, 1].sum())
    tgt = d[d[:, 5] == 1]
    miss = int((d[:, 2] - d[:, 3]).sum())
    out["decode_tokens"] = tokens
    out["misses_per_decode_token"] = miss / tokens if tokens else None
    vs = np.array(stats.verify_steps, np.int64).reshape(-1, 2)
    out["verify_rows"] = int(vs[:, 0].sum())
    out["misses_per_verify_row"] = vs[:, 1].sum() / vs[:, 0].sum() if len(vs) else None
    out["draft_hit_rate"] = stats.draft[1] / stats.draft[0] if stats.draft[0] else None
    if len(p):
        byt = (p[:, 1] - p[:, 2]) * EXPERT_BYTES
        out["prefill_steps"] = len(p)
        out["prefill_rows"] = int(p[:, 0].sum())
        out["prefill_gb_per_step_mean"] = float(byt.mean() / 1e9)
        out["prefill_gb_per_step_max"] = float(byt.max() / 1e9)
        out["prefill_rows_per_step_mean"] = float(p[:, 0].mean())
        out["prefill_gb_per_1k_tokens"] = float(byt.sum() / 1e9 / (p[:, 0].sum() / 1000))
        out["prefill_hit_rate"] = float(p[:, 2].sum() / p[:, 1].sum())
    # warm-up: hit rate by decode tokens since the start of the session
    wu = []
    lo = 0
    for hi in WARM_EDGES + [10**12]:
        m = (tgt[:, 4] > lo) & (tgt[:, 4] <= hi) if len(tgt) else np.zeros(0, bool)
        if m.any():
            wu.append((lo, hi, float(tgt[m, 3].sum() / tgt[m, 2].sum()), int(tgt[m, 1].sum())))
        lo = hi
    out["warmup"] = wu
    dl = stats.dec_layer
    out["per_layer_decode_hit"] = [float(dl[l, 1] / dl[l, 0]) if dl[l, 0] else None for l in range(n_layers)]
    return out


def make_policy(name, cap, trace, profile, drafter_counts):
    nl, E = trace.n_total_layers, trace.n_experts
    counts = np.zeros((nl, E), np.int64)
    counts[: min(len(profile), nl)] = profile[:nl]
    if name == "static":
        if drafter_counts is not None:
            counts[trace.n_layers:] = drafter_counts
        return Static(cap, nl, E, counts)
    if name == "lru":
        return LRU(cap, nl, E)
    if name == "gufo":
        return Gufo(cap, nl, E, counts)
    if name == "hotness":
        return Hotness(cap, nl, E)
    raise ValueError(name)


def simulate(trace, profile, sizes_gb, policies, rechunks, warm=None, log=None):
    drafter = trace.histogram(drafter=True) if trace.n_mtp and any(r.kind == 1 for r in trace.records) else None
    res = []
    for gb in sizes_gb:
        cap = int(gb * 1e9 / EXPERT_BYTES)
        for pol in policies:
            for rc in rechunks:
                P = make_policy(pol, cap, trace, profile, drafter)
                if warm is not None:
                    run(warm, P, rc, count=False)
                st = run(trace, P, rc)
                r = report(st, pol, trace.n_total_layers)
                r.update(policy=pol, size_gb=gb, capacity_experts=cap, rechunk=rc)
                res.append(r)
                if log:
                    log(r)
    return res


def fmt(x, f="{:.3f}"):
    return "-" if x is None else f.format(x)


def print_table(name, res, out=sys.stdout):
    print(f"\n== {name}", file=out)
    print(f"{'GB':>5} {'policy':<8} {'chunk':>6} {'dec hit%':>8} {'miss/tok':>8} {'miss/vrow':>9} {'drft hit%':>9} "
          f"{'pre hit%':>8} {'pre GB/step':>11} {'GB/1k tok':>9}", file=out)
    for r in res:
        print(f"{r['size_gb']:>5g} {r['policy']:<8} {r['rechunk'] or 'rec':>6} "
              f"{fmt(None if r['decode_hit_rate'] is None else 100 * r['decode_hit_rate'], '{:.2f}'):>8} "
              f"{fmt(r['misses_per_decode_token'], '{:.2f}'):>8} {fmt(r['misses_per_verify_row'], '{:.2f}'):>9} "
              f"{fmt(None if r['draft_hit_rate'] is None else 100 * r['draft_hit_rate'], '{:.2f}'):>9} "
              f"{fmt(None if r.get('prefill_hit_rate') is None else 100 * r['prefill_hit_rate'], '{:.1f}'):>8} "
              f"{fmt(r.get('prefill_gb_per_step_mean'), '{:.1f}'):>11} {fmt(r.get('prefill_gb_per_1k_tokens'), '{:.2f}'):>9}",
              file=out)


def print_extras(res, out=sys.stdout):
    print("\nwarm-up (decode hit% by committed decode tokens since the start; recorded steps):", file=out)
    for r in res:
        if r["rechunk"]:
            continue
        w = " ".join(f"({lo}-{hi if hi < 10**11 else 'end'}]:{100 * h:.1f}" for lo, hi, h, _ in r["warmup"])
        print(f"  {r['size_gb']:>5g} GB {r['policy']:<8} {w}", file=out)
    print("\nper-layer decode hit% (recorded steps, layers 0,1,2,10,20,30,40,42[,43-45 drafter]):", file=out)
    for r in res:
        if r["rechunk"]:
            continue
        ls = [l for l in (0, 1, 2, 10, 20, 30, 40, 42, 43, 44, 45) if l < len(r["per_layer_decode_hit"])]
        v = " ".join(f"L{l}:{fmt(None if r['per_layer_decode_hit'][l] is None else 100 * r['per_layer_decode_hit'][l], '{:.1f}')}"
                     for l in ls)
        print(f"  {r['size_gb']:>5g} GB {r['policy']:<8} {v}", file=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--profile", default=os.path.join(_here, "..", "..", "data", "expert_profile_unholy_mixed.txt"))
    ap.add_argument("--sizes", type=float, nargs="+", default=[80, 95, 110], help="cache size, GB")
    ap.add_argument("--policies", nargs="+", default=["static", "lru", "gufo", "hotness"])
    ap.add_argument("--rechunk", type=int, nargs="*", default=[1024, 2048, 4096, 8192],
                    help="also replay with prefill regrouped in steps of this many tokens")
    ap.add_argument("--warm", help="a trace replayed first (not counted) to warm the cache")
    ap.add_argument("--json", help="write all results here")
    a = ap.parse_args()
    profile = rt.read_profile(a.profile)
    warm = rt.open_trace(a.warm) if a.warm else None
    allres = {}
    for path in a.traces:
        t = rt.open_trace(path)
        name = os.path.basename(path)
        res = simulate(t, profile, a.sizes, a.policies, [0] + a.rechunk, warm)
        allres[name] = res
        print_table(name, res)
        print_extras(res)
    if a.json:
        json.dump(allres, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

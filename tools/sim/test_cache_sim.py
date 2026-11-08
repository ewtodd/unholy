#!/usr/bin/env python3
"""Unit tests of cache_sim.py on synthetic traces whose answers are worked out by hand.
   python3 -m unittest tools/sim/test_cache_sim.py   (numpy needed)"""
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sim as cs  # noqa: E402

rt = cs.rt


def write_trace(path, steps, n_layers=1, n_mtp=0, k=1, n_experts=8):
    """steps: list of (kind, step_no, rows [(seq, pos, slot, row_kind)], ids [layers][rows][k] nested lists)."""
    with open(path, "wb") as f:
        f.write(b"UHROUTE1" + struct.pack("<4H", n_layers, n_mtp, k, n_experts))
        for kind, step, rows, ids in steps:
            ids = np.array(ids, np.uint8)
            layer0 = 0 if kind == 0 else n_layers
            f.write(rt.HEADER.pack(rt.MAGIC, kind, 0, ids.shape[0], layer0, k, 0, step, len(rows), 0))
            f.write(np.array([tuple(r) + (0,) for r in rows], dtype=rt.ROW).tobytes())
            f.write(ids.tobytes())


def decode_steps(picks, seq=1, pos0=10):
    """One decode row per step, one layer, k = 1."""
    return [(0, i, [(seq, pos0 + i, 0, 1)], [[[e]]]) for i, e in enumerate(picks)]


def sim(steps, policy, cap, profile=None, rechunk=0, **kw):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.bin")
        write_trace(p, steps, **kw)
        t = rt.open_trace(p)
        counts = np.zeros((t.n_total_layers, t.n_experts), np.int64) if profile is None else profile
        P = cs.make_policy(policy, cap, t, counts, None)
        st = cs.run(t, P, rechunk)
        return cs.report(st, policy, t.n_total_layers), P, t


class Tests(unittest.TestCase):
    def test_roundtrip_histogram(self):
        steps = decode_steps([0, 1, 1, 7])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.bin")
            write_trace(p, steps)
            t = rt.open_trace(p)
            self.assertEqual(len(t.records), 4)
            self.assertEqual(t.histogram()[0].tolist(), [1, 2, 0, 0, 0, 0, 0, 1])

    def test_static(self):
        # profile: experts 0 and 1 hottest; cache 2 -> {0, 1}. picks 0 1 0 2 3 1: hits at 0, 1, 0, 1 = 4 of 6
        prof = np.array([[10, 9, 1, 1, 0, 0, 0, 0]], np.int64)
        r, _, _ = sim(decode_steps([0, 1, 0, 2, 3, 1]), "static", 2, prof)
        self.assertAlmostEqual(r["decode_hit_rate"], 4 / 6)
        self.assertEqual(r["decode_tokens"], 6)
        self.assertAlmostEqual(r["misses_per_decode_token"], 2 / 6)

    def test_lru(self):
        # cap 2, picks 0 1 0 2 1 3: miss miss hit miss(evict 1) miss(evict 0) miss(evict 2) -> 1 hit of 6
        r, P, _ = sim(decode_steps([0, 1, 0, 2, 1, 3]), "lru", 2)
        self.assertAlmostEqual(r["decode_hit_rate"], 1 / 6)
        self.assertAlmostEqual(r["misses_per_decode_token"], 5 / 6)
        self.assertEqual(sorted(np.flatnonzero(P.inc).tolist()), [1, 3])

    def test_lru_never_evicts_the_batch(self):
        # k = 2, cap 2: step 1 uses {0, 1}; step 2 uses {2, 3}: both must come in, evicting 0 and 1
        steps = [(0, 0, [(1, 0, 0, 1)], [[[0, 1]]]), (0, 1, [(1, 1, 0, 1)], [[[2, 3]]])]
        r, P, _ = sim(steps, "lru", 2, k=2)
        self.assertEqual(sorted(np.flatnonzero(P.inc).tolist()), [2, 3])
        self.assertEqual(r["decode_hit_rate"], 0.0)

    def test_hotness(self):
        # cap 2, picks 0 0 0 1 2: hot[0] = 3 stays, 1 is evicted for 2; hits at the 2nd and 3rd pick
        r, P, _ = sim(decode_steps([0, 0, 0, 1, 2]), "hotness", 2)
        self.assertAlmostEqual(r["decode_hit_rate"], 2 / 5)
        self.assertEqual(sorted(np.flatnonzero(P.inc).tolist()), [0, 2])
        # halving every 16 decode tokens
        H = cs.Hotness(4, 1, 8)
        H.hot[:] = [16, 7, 3, 1, 0, 0, 0, 0]
        H.note_decode(15)
        self.assertEqual(H.hot[:4].tolist(), [16, 7, 3, 1])
        H.note_decode(1)
        self.assertEqual(H.hot[:4].tolist(), [8, 3, 1, 0])
        H.note_decode(32)
        self.assertEqual(H.hot[:4].tolist(), [2, 0, 0, 0])

    def test_gufo_seed_and_admission(self):
        E = 32
        prof = np.zeros((1, E), np.int64)
        prof[0, :15] = 100 - np.arange(15)
        G = cs.Gufo(16, 1, E, prof)  # 15/16 of 16 = 15 experts seeded, one slot free
        self.assertEqual(int(G.inc.sum()), 15)
        self.assertAlmostEqual(G.score[0], min(8.0, np.log2(101.0)))
        # a 20-row step: expert 20 x3 rows and 21 x1 row (both outside the last 16 rows), expert 0 the rest
        ex = np.array([0, 20, 21])
        cnt = np.array([16, 3, 1])
        l16 = np.array([True, False, False])
        hit = G.access(0, ex, cnt, l16, 20)
        self.assertEqual(hit.tolist(), [True, False, False])
        self.assertTrue(G.inc[20])        # took the free slot
        self.assertFalse(G.inc[21])       # full now: 20 rows > 16, not in the last 16, count 1 < max(4, 20/32)
        # same with 3 rows: admitted at once, evicting the lowest score outside the batch (expert 20: score 3 < log2(87))
        hit = G.access(0, np.array([21]), np.array([1]), np.array([True]), 3)
        self.assertFalse(hit[0])
        self.assertTrue(G.inc[21])
        self.assertFalse(G.inc[20])
        self.assertEqual(int(G.inc.sum()), 16)
        # an expert in the last 16 rows of a big step is admitted, one with count >= max(4, n/32) too
        G.access(0, np.array([22, 23]), np.array([1, 40]), np.array([True, False]), 1280)
        self.assertTrue(G.inc[22] and G.inc[23])
        self.assertEqual(int(G.inc.sum()), 16)

    def test_gufo_decay(self):
        G = cs.Gufo(4, 1, 8, np.zeros((1, 8), np.int64))
        G.inc[0] = True
        G.n = 1
        G.score[0], G.stamp[0], G.now = 8.0, 0, 1600
        self.assertAlmostEqual(float(G.eff(np.array([0]))[0]), 4.0)  # 8 / (1 + 1600/1600)

    def test_committed_and_verify(self):
        # seq 1: verify chunk of 6 rows at 10, next chunk at 13 (3 accepted), then a decode row at 17 (4 committed
        # by the chunk at 13: 17 - 13 = 4 but at most its 6 rows), the last chunk counts 1
        rows = lambda s, p0, n, kd: [(s, p0 + i, 0, kd) for i in range(n)]
        steps = [(0, 0, rows(1, 10, 6, 2), [[[0]] * 6]),
                 (0, 1, rows(1, 13, 6, 2), [[[1]] * 6]),
                 (0, 2, rows(1, 17, 1, 1), [[[2]]])]
        r, _, t = sim(steps, "lru", 8)
        b = list(cs.batches_of(t))
        self.assertEqual([x.committed for x in b], [3, 4, 1])
        self.assertEqual(r["decode_tokens"], 8)
        self.assertEqual(r["verify_rows"], 12)
        self.assertAlmostEqual(r["misses_per_verify_row"], 2 / 12)  # step 0: 1 distinct expert miss, step 1 likewise

    def test_rechunk_regroups_prefill(self):
        rows = [(1, i, 0, 0) for i in range(10)]
        ids = [[[i % 8] for i in range(10)]]
        steps = [(0, 0, rows, ids), (0, 1, [(1, 10, 0, 1)], [[[3]]])]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.bin")
            write_trace(p, steps)
            t = rt.open_trace(p)
            rec = list(cs.batches_of(t))
            self.assertEqual([(b.kind, b.rows) for b in rec], [("prefill", 10), ("decode", 1)])
            rc = list(cs.batches_of(t, 4))
            self.assertEqual([(b.kind, b.rows) for b in rc], [("prefill", 4), ("prefill", 4), ("prefill", 2), ("decode", 1)])
            self.assertEqual(np.concatenate([b.ids[0, :, 0] for b in rc[:3]]).tolist(), [i % 8 for i in range(10)])
            # bytes read with an unbounded LRU: 8 distinct experts either way, but per step the reads split:
            # whole step: reads 8 (all miss); 4-row chunks: reads 4 + 4 + 2, misses 4 + 4 + 0
            r0, _, _ = sim(steps, "lru", 8)
            r4, _, _ = sim(steps, "lru", 8, rechunk=4)
            self.assertEqual(r0["prefill_steps"], 1)
            self.assertAlmostEqual(r0["prefill_gb_per_step_mean"], 8 * cs.EXPERT_BYTES / 1e9)
            self.assertEqual(r4["prefill_steps"], 3)
            self.assertAlmostEqual(r4["prefill_gb_per_step_mean"], 8 * cs.EXPERT_BYTES / 1e9 / 3)
            self.assertAlmostEqual(r4["prefill_gb_per_step_max"], 4 * cs.EXPERT_BYTES / 1e9)

    def test_drafter_layers_share_the_cache(self):
        # 1 target layer + 1 drafter layer, cap 1: the drafter's expert 0 evicts the target's expert 0
        steps = [(0, 0, [(1, 0, 0, 1)], [[[0]]]), (1, 0, [(0, 1, 0, 3)], [[[0]]]), (0, 1, [(1, 1, 0, 1)], [[[0]]])]
        r, _, _ = sim(steps, "lru", 1, n_mtp=1)
        self.assertEqual(r["decode_hit_rate"], 0.0)
        self.assertEqual(r["draft_hit_rate"], 0.0)

    def test_warmup_buckets(self):
        r, _, _ = sim(decode_steps([0] * 40), "lru", 2)
        # first 32 tokens: 1 miss in 32 reads; the next 8: all hits
        (lo0, hi0, h0, n0), (lo1, hi1, h1, n1) = r["warmup"][:2]
        self.assertAlmostEqual(h0, 31 / 32)
        self.assertAlmostEqual(h1, 1.0)
        self.assertEqual((n0, n1), (32, 8))


if __name__ == "__main__":
    unittest.main()

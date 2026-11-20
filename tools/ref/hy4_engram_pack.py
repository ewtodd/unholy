"""Pack Hy4 flash's Engram tables for unholy: e4m3 codes + one bf16 scale per row, and the hash parameters.

    /work/env/hy4/bin/python tools/ref/hy4_engram_pack.py --ckpt /scratch/models/hy4-flash-fp8 \
        --out /scratch/models/hy4-flash-engram

The release keeps each Engram layer's table in bf16 ([rows, 96], 31.7 GB per layer). transformers reads a row as
_fp8_round_trip(row): s = bf16(amax(|row|) / 448), q = e4m3(bf16(row / s)), value = bf16(float(q) * s) (the
straight-through `x + (q - x)` is exact: q - x is a bf16 by Sterbenz). Storing q and s therefore loses nothing the
model reads, at 97 bytes a row instead of 192.

Per Engram layer L (the decoder layer that holds it; its hash seed uses layer id L + 1) the output has
`layer_L.q` (uint8 [rows, 96]), `layer_L.s` (bf16 [rows]) and, in meta.json, the n-gram hash: the compressed-id map
(lookup_table, also as `lookup.i32`, identical for every layer in the release but stored per layer), the odd
multipliers (numpy's default_rng(engram_seed + 10007 * (L + 1)), as transformers' draw_hash_multipliers), each
head's modulus (its table size) and row offset.
"""
import argparse
import json
import os
import struct
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-per-chunk", type=int, default=1 << 22)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.ckpt, "config.json")) as f:
        tc = json.load(f)["text_config"]
    with open(os.path.join(args.ckpt, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]

    def tensor(name):
        fn = os.path.join(args.ckpt, wm[name])
        with open(fn, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))[name]
        b, e = hdr["data_offsets"]
        return np.memmap(fn, dtype=np.uint8, mode="r", offset=8 + n + b, shape=(e - b,)), hdr

    n = tc["engram_max_ngram_size"]
    heads = tc["engram_n_head_per_ngram"]
    layers = sorted({int(k.split(".")[2]) for k in wm if ".engram_layer." in k})
    meta = {"source": os.path.abspath(args.ckpt), "max_ngram": n, "heads_per_ngram": heads,
            "head_dim": tc["engram_n_embed_per_ngram"] // heads, "pad_id": tc["engram_pad_id"], "layers": {}}
    fmax = torch.finfo(torch.float8_e4m3fn).max
    for L in layers:
        lid = L + 1  # transformers' HYV4FlashEngramLayer(config, layer_idx + 1)
        sizes = [s for order in tc["engram_layer_head_vocab_sizes"][str(lid)] for s in order]
        lt_raw, lt_hdr = tensor(f"model.layers.{L}.engram_layer.lookup_table")
        lookup = lt_raw.view(np.int32).copy()
        compressed = int(lookup.max()) + 1
        half = max(1, np.iinfo(np.int64).max // compressed // 2)
        rng = np.random.default_rng(tc["engram_seed"] + 10007 * lid)
        mult = (rng.integers(0, half, size=n, dtype=np.int64) * 2 + 1).tolist()
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).tolist()
        lookup.tofile(os.path.join(args.out, f"layer_{L}.lookup.i32"))

        raw, hdr = tensor(f"model.layers.{L}.engram_layer.multi_head_embedding.embedding.weight")
        rows, dim = hdr["shape"]
        if hdr["dtype"] != "BF16" or rows != sum(sizes):
            raise SystemExit(f"layer {L}: unexpected table {hdr['dtype']} {hdr['shape']} vs {sum(sizes)} rows")
        tab = raw.view(np.uint16).reshape(rows, dim)
        qf = open(os.path.join(args.out, f"layer_{L}.q"), "wb")
        sf = open(os.path.join(args.out, f"layer_{L}.s"), "wb")
        t0 = time.time()
        for r0 in range(0, rows, args.rows_per_chunk):
            x = torch.from_numpy(np.array(tab[r0:r0 + args.rows_per_chunk])).view(torch.bfloat16)
            s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fmax  # bf16, as _fp8_round_trip
            q = (x / s).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
            if r0 == 0:  # the round trip from (q, s) equals transformers' on this chunk
                ref = (x / s).clamp(-fmax, fmax).to(torch.float8_e4m3fn).to(x.dtype) * s
                ref = x + (ref - x)
                got = q.to(torch.bfloat16) * s
                bad = (ref.float() != got.float()).sum().item()  # values (the straight-through turns -0 into +0)
                if bad:
                    i = (ref.float() != got.float()).nonzero()[:4].tolist()
                    raise SystemExit(f"layer {L}: (q, s) does not reproduce the round trip at {bad} values, e.g. "
                                     + ", ".join(f"{x[r, c].item()} -> {ref[r, c].item()} vs {got[r, c].item()}"
                                                 for r, c in i))
            qf.write(q.view(torch.uint8).numpy().tobytes())
            sf.write(s.view(torch.int16).numpy().tobytes())
            if (r0 // args.rows_per_chunk) % 8 == 0:
                print(f"layer {L}: {r0 + x.shape[0]}/{rows} rows, {time.time() - t0:.0f} s", flush=True)
        qf.close()
        sf.close()
        meta["layers"][str(L)] = {"layer_id": lid, "rows": rows, "multipliers": [str(m) for m in mult],
                                  "moduli": sizes, "offsets": offsets, "compressed_vocab": compressed,
                                  "lookup_size": int(lookup.shape[0])}
        print(f"layer {L}: done in {time.time() - t0:.0f} s; multipliers {mult}", flush=True)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()

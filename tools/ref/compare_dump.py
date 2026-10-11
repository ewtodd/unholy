"""Compare engine dumps (UNHOLY_DUMP=dir, files L###_<name>.bin) with oracle captures
(run.py --capture-layer L, files cap_###/<name>.bin).

    python3 tools/ref/compare_dump.py <engine_dump_dir> <oracle_cap_dir> <layer>
"""
import os
import sys

import numpy as np

# name -> (dtype, note); bf16 files are uint16.
KINDS = {
    "post": np.float32, "comb": np.float32, "gate_out0": np.float32,
    "gate_out1": None, "indexer_out": None,
}


def load(path, name):
    raw = np.fromfile(path, dtype=np.uint8)
    for key, dt in KINDS.items():
        if name.endswith(key):
            if dt is None:  # int: engine int32, oracle int64
                return raw.view(np.int64 if raw.size % 8 == 0 and "cap_" in path else np.int32).astype(np.int64)
            return raw.view(dt).astype(np.float64)
    u = raw.view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).astype(np.float64)


def main():
    edir, odir, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
    names = sorted(f[:-4] for f in os.listdir(odir) if f.endswith(".bin"))
    for name in names:
        ep = os.path.join(edir, f"L{layer:03d}_{name}.bin")
        if not os.path.exists(ep):
            print(f"{name:20s} (no engine dump)")
            continue
        a, b = load(ep, name), load(os.path.join(odir, name + ".bin"), name)
        if a.size != b.size:
            print(f"{name:20s} size mismatch engine {a.size} oracle {b.size}")
            continue
        if name.endswith(("gate_out1", "indexer_out")):
            print(f"{name:20s} int: {np.mean(a == b) * 100:6.2f}% equal")
            continue
        d = np.abs(a - b)
        rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        print(f"{name:20s} rel L2 {rel:.3e}  max|d| {d.max():.3e}  identical {np.mean(a == b) * 100:6.2f}%")


if __name__ == "__main__":
    main()

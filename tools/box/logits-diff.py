"""Two f32 logits dumps of the same positions (unholy-hy4-resident-test --out, the oracle's logits.bin): max |diff|,
KL and top-1 agreement, split into the prefill rows and the last N (decode) rows.

    python3 tools/box/logits-diff.py A.bin B.bin DECODE_ROWS [VOCAB]
"""
import sys

import numpy as np


def main():
    a_path, b_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    V = int(sys.argv[4]) if len(sys.argv) > 4 else 120832
    a = np.fromfile(a_path, dtype=np.float32).reshape(-1, V)
    b = np.fromfile(b_path, dtype=np.float32).reshape(-1, V)

    def lsm(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    for name, sl in (("prefill", slice(0, a.shape[0] - n)), ("decode", slice(a.shape[0] - n, None))):
        if sl.stop is not None and sl.stop <= 0 or a[sl].shape[0] == 0:
            continue
        la, lb = lsm(a[sl].astype(np.float64)), lsm(b[sl].astype(np.float64))
        kl = (np.exp(la) * (la - lb)).sum(-1)
        print(f"{name:7s} rows {la.shape[0]:5d}  max |diff| {float(np.abs(a[sl] - b[sl]).max()):.3g}  mean KL "
              f"{float(kl.mean()):.3e}  top-1 same {float((a[sl].argmax(-1) == b[sl].argmax(-1)).mean()):.4f}")


if __name__ == "__main__":
    main()

"""KL(a || b) and top-1 agreement between two f32 logits files [S, V].

    python3 tools/ref/kl.py a/logits.bin b/logits.bin [vocab]
"""
import sys

import numpy as np


def main():
    V = int(sys.argv[3]) if len(sys.argv) > 3 else 129280
    a = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, V).astype(np.float64)
    b = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, V).astype(np.float64)
    la = a - (np.log(np.exp(a - a.max(1, keepdims=True)).sum(1, keepdims=True)) + a.max(1, keepdims=True))
    lb = b - (np.log(np.exp(b - b.max(1, keepdims=True)).sum(1, keepdims=True)) + b.max(1, keepdims=True))
    kl = (np.exp(la) * (la - lb)).sum(1)
    agree = (a.argmax(1) == b.argmax(1)).mean()
    print(f"{len(kl)} positions: mean KL {kl.mean():.3e}  median {np.median(kl):.3e}  "
          f"p99 {np.quantile(kl, 0.99):.3e}  max {kl.max():.3e}  top-1 agreement {agree * 100:.2f}%")


if __name__ == "__main__":
    main()

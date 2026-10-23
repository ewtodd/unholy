#!/usr/bin/env bash
# One-time inputs on son-of-anton, each made only when missing:
#   /work/env/py               python with ROCm torch (a GC root to a pinned env, see below)
#   $IDS                       32768 code-text token ids (mkids.py)
#   $BOX_DATA/gsm48.jsonl      the engine-batch request set (mkgsm.py)
# The oracle and the baselines need the GPUs: tools/box/oracle.sh, then validate.sh --baseline.
set -euo pipefail
. "$(dirname "$0")/env.sh"
mkdir -p "$BOX_DATA" "$BOX_RUNS" /work/env

# The env has numpy, tokenizers, torch (ROCm), safetensors, pyarrow. Building torch from nixpkgs is not in
# the binary cache, so pin the store path that already exists rather than re-evaluating env.nix.
: "${PYENV_STORE:=/nix/store/mi72g0np7779ff5l70mfbm6xfwhp6k9k-python3-3.14.7-env}"
if [ ! -x /work/env/py ]; then
  nix-store --add-root /work/env/pyenv --realise "$PYENV_STORE" >/dev/null
  printf '#!/bin/sh\nexport OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\nexec /work/env/pyenv/bin/python3 "$@"\n' > /work/env/py
  chmod +x /work/env/py
fi

[ -f "$IDS" ] || "$REF_PY" "$BOX_TREE/tools/box/mkids.py" --tokenizer "$MODEL/tokenizer.json" --src "$BOX_TREE" \
  --corpus "$BOX_DATA/code-corpus.txt" --out "$IDS"
[ -f "$BOX_DATA/gsm48.jsonl" ] || "$REF_PY" "$BOX_TREE/tools/box/mkgsm.py" --out "$BOX_DATA/gsm48.jsonl"
ls -la "$BOX_DATA"

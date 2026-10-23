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

# GLM: transformers main (glm5_next) in a venv over the pinned env's torch, and the same corpus through GLM's
# tokenizer, starting the way its chat template does.
if [ ! -x "$GLM_PY" ]; then
  /work/env/pyenv/bin/python3 -m venv --system-site-packages "$(dirname "$(dirname "$GLM_PY")")"
  "$GLM_PY" -m pip install -q "transformers @ https://codeload.github.com/huggingface/transformers/tar.gz/$TRANSFORMERS_REV"
fi
if [ ! -f "$GLM_IDS" ] && [ -f "$GLM_MODEL/tokenizer.json" ]; then
  mkdir -p "$(dirname "$GLM_IDS")"
  "$REF_PY" "$BOX_TREE/tools/box/mkids.py" --tokenizer "$GLM_MODEL/tokenizer.json" --src "$BOX_TREE" \
    --corpus "$BOX_DATA/code-corpus.txt" --out "$GLM_IDS" --bos -1 --prefix '[gMASK]<sop>'
fi
ls -la "$BOX_DATA"

#!/usr/bin/env bash
# The DSV4 oracle on the code prompt: DeepSeek's model.py layer-streamed on one R9700 (tools/ref/oracle),
# 1024-token prefill + 8 teacher-forced decode steps, logits and per-layer hidden states.
#   tools/box/oracle.sh [OUT-DIR]      (default $ORACLE)
set -euo pipefail
. "$(dirname "$0")/env.sh"
out=${1:-$ORACLE}
mkdir -p "$out"
with_gpus env PYTHONPATH="$BOX_TREE/tools/ref/oracle:$MODEL/inference" \
  "$REF_PY" "$BOX_TREE/tools/ref/oracle/run.py" --ckpt "$MODEL" --ids "$IDS" --prompt 1024 --decode 8 \
  --out "$out" 2>&1 | tee "$out/oracle.log"

#!/usr/bin/env bash
# The GLM-5.3-Flash oracle on the code prompt: transformers' glm5_next on the FP8 checkpoint, layer-streamed
# on one R9700 (tools/ref/glm_oracle), 1024-token prefill + 8 teacher-forced decode steps.
#   tools/box/glm-oracle.sh [OUT-DIR] [run.py flags, e.g. --layers 4]      (default OUT-DIR: $GLM_ORACLE)
set -euo pipefail
. "$(dirname "$0")/env.sh"
out=${1:-$GLM_ORACLE}; shift || true
mkdir -p "$out"
with_gpus "$GLM_PY" "$BOX_TREE/tools/ref/glm_oracle/run.py" --ckpt "$GLM_MODEL" --ids "$GLM_IDS" --prompt 1024 \
  --decode 8 --out "$out" "$@" 2>&1 | tee "$out/oracle.log"

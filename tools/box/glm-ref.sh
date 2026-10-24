#!/usr/bin/env bash
# GLM-5.3-Flash reference forward against the GLM oracle (unholy-glm-ref), layer-streamed on card 0.
#   tools/box/glm-ref.sh LABEL [unholy-glm-ref flags, e.g. --isolate --layers 5, --dense DIR --experts DIR]
# The logits of a full run go to the run directory (logits.bin).
set -euo pipefail
. "$(dirname "$0")/env.sh"
label=${1:?label}; shift
d=$(run_dir "glm-ref-$label")
with_gpus in_nix "$BUILD/unholy-glm-ref" --ckpt "$GLM_MODEL" --oracle "$GLM_ORACLE" --out "$d/logits.bin" "$@" 2>&1 | tee "$d/log.txt"

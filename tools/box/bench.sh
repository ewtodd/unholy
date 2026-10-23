#!/usr/bin/env bash
# Prefill / decode microbench: unholy-bench on the code ids.
#   tools/box/bench.sh LABEL [unholy-bench flags, e.g. --ctx 16384 --batch 8 --steps 64]
set -euo pipefail
. "$(dirname "$0")/env.sh"
label=${1:?label}; shift
d=$(run_dir "bench-$label")
with_gpus in_nix "$BUILD/unholy-bench" --model "$MODEL" --profile "$PROFILE" --ids "$IDS" "$@" 2>&1 | tee "$d/bench.txt"

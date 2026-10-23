#!/usr/bin/env bash
# Depth curve: unholy-curve, one sequence to 512k in 4096-token chunks with two in flight, decode probes at
# 4k / 32k / 128k / 512k (compare with docs/bench/curve-*).
#   tools/box/curve.sh LABEL [unholy-curve flags]
set -euo pipefail
. "$(dirname "$0")/env.sh"
label=${1:?label}; shift
d=$(run_dir "curve-$label")
with_gpus in_nix "$BUILD/unholy-curve" --model "$MODEL" --profile "$PROFILE" --ids "$IDS" "$@" 2>&1 | tee "$d/curve.txt"

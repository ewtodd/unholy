#!/usr/bin/env bash
# A/B decode timing: unholy-curve from two builds, alternating, ROUNDS rounds (single runs vary by ~0.3 ms/token).
#   tools/box/ab.sh BUILD_A BUILD_B [ROUNDS [unholy-curve flags]]     default: 4k context, 128 decode steps
set -euo pipefail
. "$(dirname "$0")/env.sh"
a=${1:?build A}; b=${2:?build B}; rounds=${3:-3}
shift $(($# < 3 ? $# : 3))
flags=("$@")
[ ${#flags[@]} -gt 0 ] || flags=(--ctx 4096 --chunk 4096 --probes 4096 --decode-steps 128)
d=$(run_dir ab)
run() { with_gpus in_nix "$1/unholy-curve" --model "$MODEL" --profile "$PROFILE" --ids "$IDS" "${flags[@]}" 2>&1 | grep "decode at"; }
for _ in $(seq "$rounds"); do
  echo "A $(run "$a")" | tee -a "$d/ab.txt"
  echo "B $(run "$b")" | tee -a "$d/ab.txt"
done

#!/usr/bin/env bash
# Resident-model regression check on the oracle prompt: unholy-resident-test's logits against the saved
# baseline (bit-identical expected when a change keeps the arithmetic) and KL / top-1 against the oracle.
#   tools/box/validate.sh [resident-test flags]               compare with $BOX_DATA/base/resident.bin
#   tools/box/validate.sh --baseline [resident-test flags]    save a new baseline
# Bit-exact comparisons need the baseline's expert placement: keep MARGIN (card margin, bytes) fixed, and
# adjust it by any change in planner scratch so the placement stays the same.
set -euo pipefail
. "$(dirname "$0")/env.sh"
: "${MARGIN:=2147483648}"
base=$BOX_DATA/base/resident.bin
save=0
if [ "${1:-}" = --baseline ]; then save=1; shift; fi
d=$(run_dir validate)
args=(--oracle "$ORACLE" --profile "$PROFILE" --card-margin-bytes "$MARGIN" --out "$d/logits.bin")
[ $save = 1 ] || args+=(--ref "$base")
with_gpus in_nix "$BUILD/unholy-resident-test" "${args[@]}" "$@" 2>&1 | tee "$d/log.txt"
if [ $save = 1 ]; then
  mkdir -p "$(dirname "$base")"
  cp "$d/logits.bin" "$base"
  echo "baseline saved: $base (MARGIN=$MARGIN, tree $(cat "$BOX_TREE/.synced-rev" 2>/dev/null))" | tee "$base.txt"
fi

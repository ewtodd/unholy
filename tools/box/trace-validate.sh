#!/usr/bin/env bash
# The routing trace must not change any output: unholy-resident-test's logits with the trace off (a fresh
# baseline of this tree) and with UNHOLY_ROUTING_TRACE set, compared bit for bit; the trace is summarized.
# Does not touch the shared baseline ($BOX_DATA/base) and does not stop any service: run it while the box is
# paused or the GPUs are otherwise free (it takes the GPU lock).
set -euo pipefail
. "$(dirname "$0")/env.sh"
export BOX_TREE BUILD
: "${MARGIN:=2147483648}"
d=$(run_dir trace-validate)
args=(--oracle "$ORACLE" --profile "$PROFILE" --card-margin-bytes "$MARGIN")
flock /work/runs/gpu.lock bash -c '
  set -e
  . '"$BOX_TREE"'/tools/box/env.sh
  d='"$d"'
  in_nix "$BUILD/unholy-resident-test" "$@" --out "$d/off.bin" 2>&1 | tee "$d/off.log"
  UNHOLY_ROUTING_TRACE="$d/trace.bin" in_nix "$BUILD/unholy-resident-test" "$@" --out "$d/on.bin" 2>&1 | tee "$d/on.log"
' _ "${args[@]}"
if cmp "$d/off.bin" "$d/on.bin"; then echo "trace-validate: logits bit-identical, trace off vs on ($d)"; else echo "trace-validate: DIFFER"; exit 1; fi
ls -l "$d/trace.bin"

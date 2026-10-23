#!/usr/bin/env bash
# Token-for-token check under concurrency: unholy-engine-batch on the GSM8K request set with the production
# flags (plus any given here), logprobs on, then a comparison with a reference run if one is given.
#   tools/box/eb.sh LABEL CONC [engine-batch flags]
#   REF=/work/runs/<dir>/eb-c4.jsonl tools/box/eb.sh LABEL 4
# A single mismatching request needs a repeat before blaming a change: rare run-to-run logit outliers occur
# in every build (DECISIONS 2026-10-15).
set -euo pipefail
. "$(dirname "$0")/env.sh"
label=${1:?label}; conc=${2:?concurrency}; shift 2
d=$(run_dir "eb-$label")
out=$d/eb-c$conc.jsonl
with_gpus in_nix "$BUILD/unholy-engine-batch" --model "$MODEL" --profile "$PROFILE" \
  --requests "$BOX_DATA/gsm48.jsonl" --out "$out" --concurrency "$conc" --logprobs 1 "${SERVE_FLAGS[@]}" "$@" \
  2>&1 | tee "$d/log.txt"
[ -n "${REF:-}" ] && "$REF_PY" "$BOX_TREE/tools/engine_batch/compare.py" "$REF" "$out" | tee "$d/compare.txt"
echo "$out"

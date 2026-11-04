#!/usr/bin/env bash
# APU routed-GEMM benchmark: unholy-mx-test's MXFP4 GEMM on the Strix Halo APU (synthetic rows, then recorded per-expert
# row counts), its hardware counters, then gufo's FP16 routed kernel (tools/box/gufo-routed.sh builds it) on the same
# counts. A failing step is logged, the rest runs. Count files: one line of rows per expert (data/apu0 on the box).
#   tools/box/apu-gemm.sh LABEL        (APU_DEV=2, COUNTS=$BOX_DATA/apu0)
set -euo pipefail
. "$(dirname "$0")/env.sh"
label=${1:?label}
APU_DEV=${APU_DEV:-2}
COUNTS=${COUNTS:-$BOX_DATA/apu0}
d=$(run_dir "apu-gemm-$label")
mx=$BUILD/unholy-mx-test
gufo=$BUILD/gufo-routed-bench
pmc_file=$COUNTS/dsv4-apu-l20-part2048.txt

# step LOG CMD...: run CMD in the tree's dev shell into $d/LOG; a failure is noted in the log and the script goes on.
step() {
  local log=$d/$1; shift
  echo "== $*" >"$log"
  local rc=0
  in_nix "$@" >>"$log" 2>&1 || rc=$?
  [ "$rc" -eq 0 ] || echo "== FAILED (exit $rc)" >>"$log"
}

run_all() {
  local r f n g c
  # 1. baseline at the synthetic row counts
  for r in 16 32 48 96; do
    step "base-rows$r.txt" "$mx" --dev "$APU_DEV" --gemm-only 1 --experts 64 --rows-mean "$r"
  done
  # 2. the count files
  for f in "$COUNTS"/*.txt; do
    n=$(basename "$f" .txt)
    step "mx-$n.txt" "$mx" --dev "$APU_DEV" --gemm-only 1 --no-ref 1 --reps 5 --counts "$f"
  done
  # 3. hardware counters (gfx1151 may not support them), three passes
  n=1
  for g in "SQ_WAVES SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_INSTS_VALU" \
           "SQ_INSTS_LDS SQ_INSTS_SALU SQ_WAIT_INST_ANY SQ_WAIT_INST_LDS" \
           "SQ_INST_CYCLES_VMEM SQC_LDS_BANK_CONFLICT GL2C_HIT GL2C_MISS"; do
    # shellcheck disable=SC2086  # the group is a word list
    step "pmc$n.txt" rocprofv3 --pmc $g --output-format csv -d "$d/pmc$n" -o run -- \
      "$mx" --dev "$APU_DEV" --gemm-only 1 --no-ref 1 --reps 3 --counts "$pmc_file"
    n=$((n + 1))
  done
  # 4. gufo's kernel: the count files, then uniform random routing
  for f in "$COUNTS"/*.txt; do
    n=$(basename "$f" .txt)
    step "gufo-$n.txt" "$gufo" --dev "$APU_DEV" --counts "$f" --type 16 --down-type 22
  done
  step "gufo-384x2048-16-22.txt" "$gufo" --dev "$APU_DEV" --experts 384 --tokens 2048 --type 16 --down-type 22
  step "gufo-384x2048-19-17.txt" "$gufo" --dev "$APU_DEV" --experts 384 --tokens 2048 --type 19 --down-type 17
  # the key result lines of every step
  for c in "$d"/*.txt; do
    [ "$(basename "$c")" = summary.txt ] && continue
    {
      echo "### $(basename "$c")"
      grep -E "TFLOPS|bit-identical|FAILED" "$c" || echo "(no result lines)"
    } >>"$d/summary.txt"
  done
}

with_gpus run_all
echo "apu-gemm: $d (summary.txt)"

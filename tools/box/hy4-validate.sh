#!/usr/bin/env bash
# Hy4 flash checks of a change set on the box (needs every GPU free): this tree against a baseline tree.
#   tools/box/hy4-validate.sh BASE_TREE OUT_DIR [stage ...]
# Stages (all by default):
#   logits   unholy-hy4-resident-test of both trees, FP8 and NVFP4 experts, on the decode probe (16 prompt + 64 decode
#            positions) and code1024 (prefill + 8 decode): tools/box/logits-diff.py of the two (0 for bit-identical
#            changes)
#   spec     DFly2 greedy, drafted first then plain, 200 tokens at 4096 context, both trees (acceptance, speed)
#   gemv     unholy-hy4-gemv-bench of both trees (the base tree needs the target added by hand), FP8 and NVFP4, on
#            the APU (device 2) and card 0: 6 experts x 1 row (a decode step's cold experts) and 16 x 6 (verify)
#   serve    unholy serve of this tree, FP8 then NVFP4 experts, drafts on, UNHOLY_HY4_HOST_TIMING=1:
#            tools/box/served-decode.sh twice each; the --routing-log round trip on the second start
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
base=${1:?baseline tree}
out=${2:?output directory}
shift 2
stages=${*:-gemv logits spec serve}
mkdir -p "$out"
cd "$here"
M=/scratch/models/hy4-flash-fp8
G=/scratch/models/hy4-flash-angelslim/Hy4-flash-SPARK-4Bit.gguf
P=data/expert_profile_hy4_mixed.txt
probe=/work/runs/2026-11-20-hy4/decode-probe
code1024=/work/data/hy4/oracle/code1024
code4096=/work/data/hy4/oracle/code4096
run() { nix develop -c "$@"; }

if [[ " $stages " == *" gemv "* ]]; then
  for d in 2 0; do
    for fmt in fp8 nvfp4; do
      for shape in "6 1" "16 6"; do
        set -- $shape
        for t in base new; do
          tree=$([ "$t" = base ] && echo "$base" || echo "$here")
          echo -n "$t: "
          (cd "$tree" && run ./build/unholy-hy4-gemv-bench --device "$d" --fmt "$fmt" --experts "$1" --rows "$2") |
            tee -a "$out/gemv.txt"
        done
      done
    done
  done
fi

if [[ " $stages " == *" logits "* ]]; then
  for fmt in fp8 nvfp4; do
    extra=()
    [ "$fmt" = nvfp4 ] && extra=(--experts-gguf "$G")
    for o in probe code1024; do
      dir=$([ "$o" = probe ] && echo "$probe" || echo "$code1024")
      dec=$([ "$o" = probe ] && echo 64 || echo 8)
      for t in base new; do
        tree=$([ "$t" = base ] && echo "$base" || echo "$here")
        (cd "$tree" && run ./build/unholy-hy4-resident-test --oracle "$dir" --chunk 512 --profile "$here/$P" \
          "${extra[@]}" --out "$out/$fmt-$o-$t.bin" > "$out/$fmt-$o-$t.log" 2>&1)
        tail -2 "$out/$fmt-$o-$t.log"
      done
      echo "== $fmt $o: new vs base"
      /work/env/py tools/box/logits-diff.py "$out/$fmt-$o-new.bin" "$out/$fmt-$o-base.bin" "$dec" | tee -a "$out/logits.txt"
      rm -f "$out/$fmt-$o-new.bin" "$out/$fmt-$o-base.bin"
    done
  done
fi

if [[ " $stages " == *" spec "* ]]; then
  for t in base new; do
    tree=$([ "$t" = base ] && echo "$base" || echo "$here")
    echo "== spec $t"
    (cd "$tree" && run ./build/unholy-hy4-resident-test --oracle "$code4096" --chunk 512 --profile "$here/$P" \
      --experts-gguf "$G" --spec 200 --spec-only draft-first 2>&1 | tail -4) | tee "$out/spec-$t.txt"
  done
fi

if [[ " $stages " == *" serve "* ]]; then
  rlog="$out/routing-log.txt"
  rm -f "$rlog"
  for fmt in fp8 nvfp4; do
    extra=()
    [ "$fmt" = nvfp4 ] && extra=(--experts-gguf "$G")
    log="$out/serve-$fmt.log"
    UNHOLY_HY4_HOST_TIMING=1 nohup nix develop -c ./build/unholy serve --model "$M" "${extra[@]}" --profile "$P" \
      --routing-log "$rlog" --max-model-len 1048576 --kv-tokens 1048576 --max-num-seqs 8 --max-batch-tokens 2048 \
      --spec-drafts 5 --port 8300 > "$log" 2>&1 < /dev/null &
    for _ in $(seq 1 120); do sleep 5; grep -q "serving\|rror" "$log" && break; done
    grep -E "card|apu|routing log|loaded|serving|rror" "$log" || true
    echo "== serve $fmt"
    tools/box/served-decode.sh "$log" 8300 600 | tee "$out/decode-$fmt.txt"
    tools/box/served-decode.sh "$log" 8300 600 | tee -a "$out/decode-$fmt.txt"
    pkill -x unholy || true
    for _ in $(seq 1 60); do pgrep -x unholy > /dev/null || break; sleep 2; done
    if pgrep -x unholy > /dev/null; then echo "unholy did not exit after SIGTERM"; exit 1; fi
    sleep 5
    grep "hy4 host per step" "$log" | tail -2 || true
    [ -f "$rlog" ] && echo "routing log: $(grep -c '^L' "$rlog") layers written"
  done
fi

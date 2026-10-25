#!/usr/bin/env bash
# The DSV4 / GLM comparison's evals (docs/compare-protocol.md, intelligence), run from e-desktop. The model is served
# on son-of-anton:8300 from the perf tree (tools/box/model-serve.sh, up until a stop file appears) and reached through
# an SSH tunnel; check-source ($CHECK_SOURCE_DIR) runs the benches. Knoll's items and sandbox stay on this machine and
# the model is local: nothing leaves the LAN. One summary JSON (+ .jsonl rows) per bench, model and seed in $OUT_DIR.
#   tools/compare/evals.sh dsv4|glm knoll,gsm8k-chat,math500,aime
# Sampling: the model cards' (temperature 1.0, top_p 0.95, no top-k); reasoning level $LEVEL (default xhigh: DSV4
# high, GLM max with unholy's templates).
set -euo pipefail
model=${1:?dsv4 or glm}; benches=${2:-knoll,gsm8k-chat,math500,aime}
: "${CHECK_SOURCE_DIR:=$HOME/Software/check-source}"
: "${OUT_DIR:=$CHECK_SOURCE_DIR/results/compare}"
: "${BOX:=son-of-anton}"
: "${LEVEL:=xhigh}"
: "${PORT:=8300}"
case $model in
  dsv4) name=dsv4-flash
        labels=(--weights "native MXFP4 experts, fp8 dense" --activations "fp8 dense (W8A8), fp8 expert inputs"
                --kv-cache "fp8 compressed rows, bf16 window") ;;
  glm) name=glm53
       labels=(--weights "EXL3 4.05 experts and dense (6-bit)" --activations "bf16, fp16 EXL3 inputs"
               --kv-cache "fp8 latent, bf16 pooled keys") ;;
  *) echo "evals.sh: unknown model $model" >&2; exit 2 ;;
esac
mkdir -p "$OUT_DIR"
stop=/work/runs/evals.stop

ssh "$BOX" "rm -f $stop; cd /work/unholy-perf && setsid nohup tools/box/model-serve.sh $model evals \
  bash -c 'until [ -e $stop ]; do sleep 15; done; rm -f $stop' >/work/runs/evals-$model.log 2>&1 </dev/null &"
ssh -N -L "$PORT:127.0.0.1:8300" "$BOX" &
tunnel=$!
# On exit: stop the server and wait until model-serve.sh is done (the mode restored), so a next run can start.
trap 'kill $tunnel 2>/dev/null; ssh "$BOX" "touch $stop; until grep -q \"^server log:\" /work/runs/evals-$model.log; do sleep 5; done"' EXIT
until curl -fs "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "\"$name\""; do
  if ssh "$BOX" "grep -qi error /work/runs/evals-$model.log"; then
    ssh "$BOX" tail -5 "/work/runs/evals-$model.log"
    exit 1
  fi
  sleep 20
done
echo "$name up on :$PORT ($(date +%T))"

cs() { (cd "$CHECK_SOURCE_DIR" && nix develop -c env PYTHONPATH=src python -m check_source "$@"); }
common=(--base-url "http://127.0.0.1:$PORT" --model "$name" "${labels[@]}" --level "$LEVEL" --temperature 1.0
        --top-p 0.95 --top-k 0)
for b in ${benches//,/ }; do
  case $b in
    knoll)
      for s in 0 1 2; do
        cs knoll --items "$HOME/knoll/items.jsonl" --repo "UM-ANSG=$HOME/knoll/UM-ANSG" \
          --flake UM-ANSG=github:ewtodd/UM-ANSG --seed $s --max-tokens 32768 --concurrency 2 "${common[@]}" \
          --out "$OUT_DIR/knoll-$model-$LEVEL-s$s.json"
      done ;;
    gsm8k-chat)
      cs gsm8k-chat --seed 0 --max-tokens 32768 --limit 200 --concurrency 8 "${common[@]}" \
        --out "$OUT_DIR/gsm8k-chat-$model-$LEVEL-s0.json" ;;
    math500)
      for s in 0 1; do
        cs math500 --seed $s --max-tokens 32768 --limit 200 --concurrency 8 "${common[@]}" \
          --out "$OUT_DIR/math500-$model-$LEVEL-s$s.json"
      done ;;
    aime)
      for s in 0 1; do
        cs aime --years 24,25,26 --seed $s --max-tokens 65536 --concurrency 8 "${common[@]}" \
          --out "$OUT_DIR/aime-$model-$LEVEL-s$s.json"
      done ;;
    *) echo "evals.sh: unknown bench $b" >&2; exit 2 ;;
  esac
done
echo "done ($(date +%T)): $OUT_DIR"

#!/usr/bin/env bash
# A short real routing trace and its histogram check, in one hold of the GPUs (flock /work/runs/gpu.lock; no
# service is stopped: run it while the box is paused / the APU is free):
#   the production server with UNHOLY_ROUTING_TRACE set, N_CHATS GSM8K chats from $BOX_DATA/gsm48.jsonl (4 at a
#   time), one DOC_CHARS-character document (code-corpus.txt, ~32k tokens) with a short answer, then
#   /unholy/routing_profile (counts since load, like the trace) and the server stopped; routing-trace.py summary
#   --compare checks the trace's per-layer histogram against it. Output: $RUN/{trace.bin,profile.txt,summary.txt}.
#   tools/box/trace-serve.sh [unholy serve flags]
set -euo pipefail
. "$(dirname "$0")/env.sh"
export BOX_TREE BUILD
: "${N_CHATS:=6}" "${DOC_CHARS:=100000}" "${PORT:=8300}"
if [ -z "${TRACE_SERVE_INNER:-}" ]; then
  RUN=$(run_dir trace-serve)
  export RUN PORT N_CHATS DOC_CHARS TRACE_SERVE_INNER=1
  flock /work/runs/gpu.lock "$0" "$@"
  nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.numpy])' -c python3 \
    "$BOX_TREE/tools/box/routing-trace.py" summary "$RUN/trace.bin" --compare "$RUN/profile.txt" | tee "$RUN/summary.txt"
  exit
fi
URL=http://127.0.0.1:$PORT
pat="[u]nholy serve --model.*--port $PORT"
UNHOLY_ROUTING_TRACE="$RUN/trace.bin" in_nix "$BUILD/unholy" serve --model "$MODEL" --profile "$PROFILE" \
  --host 127.0.0.1 --port "$PORT" --served-model-name dsv4-flash "${SERVE_FLAGS[@]}" "$@" >"$RUN/server.log" 2>&1 &
trap 'stop_server "$pat" || true' EXIT
for _ in $(seq 1 600); do curl -sf "$URL/health" >/dev/null && break; sleep 2; done
curl -sf "$URL/health" >/dev/null
mkdir -p "$RUN/req"
i=0
while IFS= read -r line; do
  jq -c '. + {model: "dsv4-flash", max_tokens: 400}' <<<"$line" >"$RUN/req/$i.json"
  i=$((i + 1))
done < <(head -n "$N_CHATS" "$BOX_DATA/gsm48.jsonl")
ls "$RUN"/req/*.json | xargs -P 4 -I{} sh -c \
  'curl -s --max-time 1800 "$0/v1/chat/completions" -H content-type:application/json -d @"$1" >"$1.out"' "$URL" {}
head -c "$DOC_CHARS" "$BOX_DATA/code-corpus.txt" |
  jq -Rs '{model: "dsv4-flash", max_tokens: 128, messages: [{role: "user", content: (. + "\n\nSummarize what this code does in two sentences.")}]}' |
  curl -s --max-time 3600 "$URL/v1/chat/completions" -H content-type:application/json -d @- | jq -c .usage
curl -s "$URL/unholy/routing_profile" >"$RUN/profile.txt"

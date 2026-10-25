#!/usr/bin/env bash
# Prefill at depth and one-stream decode (docs/compare-protocol.md, performance), run as model-serve.sh's CMD: for
# each document $DOCS/doc-DEPTH.txt (unique Pile text per depth, tools/box/mklong.py), one request on it (time to
# first token with nothing cached), then REPEATS - 1 more with other questions (the document from the prefix cache:
# their decode); max_tokens 256, temperature 1.0, top_p 0.95. The server's request lines carry prompt tokens, TTFT,
# prefill and decode tok/s: $RUN/perf.txt.
#   DEPTHS="8192 32768" tools/box/model-serve.sh glm perf bash tools/box/perf-depth.sh
set -euo pipefail
: "${DOCS:=/work/data/compare}"
qs=("Summarize the text above in three sentences." "List five named entities that appear above."
    "What is the main topic of the last part above?" "Write one question the text above answers."
    "Give the text above a short title.")
for depth in ${DEPTHS:-8192 32768 131072 262144 524288}; do
  for i in $(seq 0 $((${REPEATS:-5} - 1))); do
    jq -n --rawfile t "$DOCS/doc-$depth.txt" --arg q "${qs[$i]}" --arg m "$MODEL_NAME" \
      '{model: $m, messages: [{role: "user", content: ($t + "\n\n" + $q)}], max_tokens: 256, temperature: 1.0,
        top_p: 0.95, seed: 7}' >"$RUN/req.json"
    curl -s --max-time 3600 "$URL/v1/chat/completions" -H content-type:application/json -d @"$RUN/req.json" \
      >"$RUN/resp-$depth-$i.json"
  done
done
grep "request " "$RUN/serve.log" | sed 's/request chatcmpl-[0-9a-f]*:/request:/' >"$RUN/perf.txt"

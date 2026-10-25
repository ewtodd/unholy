#!/usr/bin/env bash
# Prefix reuse check, run as glm-serve.sh's CMD: requests A then B share the first CHARS characters of a long prompt
# (default 40000 of /work/data/glm/long_32k.json's, ~11k tokens) and differ in the question after it, then A again
# (A2); greedy, MAXTOK tokens each, with logprobs. Replies to $RUN/pair_{A,B,A2}.json; B's and A2's should match a run
# with --prefix-snapshots 0 (same geometry: the expert placement follows it).
#   tools/box/glm-serve.sh prefix bash tools/box/prefix-pair.sh
set -euo pipefail
src=${PAIR_SRC:-/work/data/glm/long_32k.json}
for t in A B A2; do
  case $t in
    A | A2) q="Which file is listed first above? Answer briefly." ;;
    B) q="Name three functions defined above." ;;
  esac
  jq --argjson chars "${CHARS:-40000}" --argjson max "${MAXTOK:-128}" --arg q "$q" --arg m "${MODEL_NAME:-glm53}" \
    '{model: $m, messages: [{role: "user", content: (.messages[0].content[0:$chars] + "\n\n" + $q)}],
      max_tokens: $max, temperature: 0, logprobs: true}' "$src" >"$RUN/pair_$t.req.json"
  curl -s --max-time 1800 "$URL/v1/chat/completions" -H content-type:application/json \
    -d @"$RUN/pair_$t.req.json" >"$RUN/pair_$t.json"
  jq -c '{usage, text: .choices[0].message.content[0:120]}' "$RUN/pair_$t.json"
done

#!/usr/bin/env bash
# Routing of a request set through the server, run as glm-serve.sh's CMD: the histogram reset, every request of
# REQUESTS (engine-batch jsonl: one chat request body per line) sent, CONC at a time, then the routing histogram
# ($RUN/profile.txt, usage-profile format: prompts and generations) and the share on the cards ($RUN/share.txt).
#   REQUESTS=/work/data/glm/gsm-train96.jsonl CONC=8 tools/box/glm-serve.sh gen bash tools/box/route-requests.sh
set -euo pipefail
: "${REQUESTS:?REQUESTS=jsonl}"
curl -s -X POST "$URL/unholy/routing_profile/reset" >/dev/null
mkdir -p "$RUN/req"
i=0
while IFS= read -r line; do
  jq -c --arg m "${MODEL_NAME:-glm53}" '. + {model: $m}' <<<"$line" >"$RUN/req/$i.json"
  i=$((i + 1))
done <"$REQUESTS"
ls "$RUN"/req/*.json | xargs -P "${CONC:-8}" -I{} sh -c \
  'curl -s --max-time 3600 "$0/v1/chat/completions" -H content-type:application/json -d @"$1" >"$1.out"' "$URL" {}
curl -s "$URL/unholy/routing_profile" >"$RUN/profile.txt"
curl -s "$URL/unholy/routing_share" >"$RUN/share.txt"
completion=$(jq -s '[.[].usage.completion_tokens] | add' "$RUN"/req/*.out)
echo "$i requests: $completion completion tokens; on cards $(head -1 "$RUN/share.txt")"

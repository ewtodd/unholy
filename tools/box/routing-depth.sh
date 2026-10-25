#!/usr/bin/env bash
# Routing share on the cards by depth, run as glm-serve.sh's CMD: prefixes of TEXT (tools/box/mklong.py's output)
# cut at the depths in TEXT.cuts, one request each (max_tokens 1); prefix caching restores the previous prefix, so
# each request computes only its new segment. The routing histogram is reset before each, then
# /unholy/routing_share read: the share of routed pairs served by card-resident experts in that depth segment;
# the segment's routing histogram goes to $RUN/profile_DEPTH.txt (usage-profile format).
#   TEXT=/work/data/glm/long-pile.txt tools/box/glm-serve.sh depth-pile bash tools/box/routing-depth.sh
set -euo pipefail
: "${TEXT:?TEXT=mklong.py output}"
prev=0
while read -r depth chars; do
  curl -s -X POST "$URL/unholy/routing_profile/reset" >/dev/null
  jq -n --rawfile t "$TEXT" --argjson n "$chars" --arg m "${MODEL_NAME:-glm53}" \
    '{model: $m, messages: [{role: "user", content: $t[0:$n]}], max_tokens: 1, temperature: 0}' >"$RUN/depth_req.json"
  usage=$(curl -s --max-time 3600 "$URL/v1/chat/completions" -H content-type:application/json \
    -d @"$RUN/depth_req.json" | jq -c '[.usage.prompt_tokens, .usage.prompt_tokens_details.cached_tokens]')
  curl -s "$URL/unholy/routing_share" >"$RUN/share_$depth.txt"
  curl -s "$URL/unholy/routing_profile" >"$RUN/profile_$depth.txt"
  echo "segment $prev-$depth: prompt/cached $usage, on cards $(head -1 "$RUN/share_$depth.txt")"
  prev=$depth
done <"$TEXT.cuts"

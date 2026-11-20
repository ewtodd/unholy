#!/usr/bin/env bash
# Single-stream decode speed of a running server: three fixed chat requests one after another (code, prose,
# reasoning), the server's own per-request lines from its log (decode tok/s, drafts accepted).
#   tools/box/served-decode.sh LOG [PORT] [MAX_TOKENS] [EXTRA_JSON]
# EXTRA_JSON is merged into each request body, e.g. '"temperature": 0'.
set -euo pipefail
log=${1:?server log}; port=${2:-8300}; max=${3:-600}; extra=${4:-}
[ -n "$extra" ] && extra=", $extra"
before=$(grep -c "request chatcmpl" "$log" || true)
for prompt in \
  "Write a Python module implementing an LRU cache with TTL expiry, with type hints and docstrings, plus pytest tests." \
  "Explain how a compressed sparse row matrix works and when to prefer it over COO, in about 400 words." \
  "A train leaves at 9:40 and travels 210 km at 84 km/h, then waits 12 minutes, then travels 95 km at 76 km/h. When does it arrive? Show your reasoning."; do
  curl -s "http://127.0.0.1:$port/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max$extra}" > /dev/null
done
sleep 1
grep "request chatcmpl" "$log" | tail -n +$((before + 1)) | sed -E 's/.*\| (prompt [0-9]+, completion [0-9]+).*(decode [0-9.]+ tok\/s)(.*)/\1 | \2\3/'

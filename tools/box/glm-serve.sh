#!/usr/bin/env bash
# GLM-5.3-Flash (build B, EXL3 4.05) served from the perf tree on port 8300 with the production geometry, a command
# run against it, the server stopped; all under one service stop (with_gpus). The server log goes to the run dir.
#   tools/box/glm-serve.sh LABEL CMD...        (CMD sees URL=http://localhost:8300 and RUN=the run dir)
#   GLM_SERVE_FLAGS="--max-num-seqs 4" tools/box/glm-serve.sh ...   (extra / overriding serve flags)
set -euo pipefail
. "$(dirname "$0")/env.sh"
label=${1:?label}; shift
d=$(run_dir "glm-serve-$label")
: "${GLM_B:=/scratch/models/glm53-flash-exl3-4.05bpw}"
body() {
  cd "$BOX_TREE"
  nix develop -c "$BUILD/unholy" serve --model "$GLM_B" --profile "$BOX_TREE/data/expert_profile_glm53_mixed.txt" \
    --port 8300 --served-model-name glm53 "${SERVE_FLAGS[@]}" ${GLM_SERVE_FLAGS:-} >"$d/serve.log" 2>&1 &
  local pid=$!
  until grep -q "^serving glm53" "$d/serve.log"; do
    kill -0 $pid 2>/dev/null || { tail -20 "$d/serve.log"; return 1; }
    sleep 2
  done
  local rc=0
  URL=http://localhost:8300 RUN=$d "$@" || rc=$?
  stop_server "unholy serve --model $GLM_B" || rc=1
  wait $pid 2>/dev/null || true
  return $rc
}
with_gpus body "$@"
echo "server log: $d/serve.log"

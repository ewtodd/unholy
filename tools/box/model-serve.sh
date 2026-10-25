#!/usr/bin/env bash
# A model served from the perf tree on port 8300 with its production configuration, a command run against it, the
# server stopped; all under one service stop (with_gpus). The server log goes to the run dir.
#   tools/box/model-serve.sh dsv4|glm LABEL CMD...   (CMD sees URL=http://localhost:8300, RUN=the run dir and
#                                                     MODEL_NAME=the served model name)
#   dsv4: $MODEL / $PROFILE, SERVE_FLAGS (the ds4 service's), as dsv4-flash; DSV4_SERVE_FLAGS appended.
#   glm:  $GLM_B / $GLM_PROFILE, SERVE_FLAGS with MTP, a 1M-token pool and 3072-token steps (the GLM service's,
#         DECISIONS 2026-10-24), as glm53; GLM_SERVE_FLAGS appended.
set -euo pipefail
. "$(dirname "$0")/env.sh"
kind=${1:?dsv4 or glm}; label=${2:?label}; shift 2
d=$(run_dir "$kind-serve-$label")
: "${GLM_B:=/scratch/models/glm53-flash-exl3-4.05bpw}"
: "${GLM_PROFILE:=$BOX_TREE/data/expert_profile_glm53_broad.txt}"
case $kind in
  dsv4) model=$MODEL; profile=$PROFILE; name=dsv4-flash; extra=(${DSV4_SERVE_FLAGS:-}) ;;
  glm) model=$GLM_B; profile=$GLM_PROFILE; name=glm53
       extra=(--kv-tokens 1048576 --max-batch-tokens 3072 ${GLM_SERVE_FLAGS:-}) ;;
  *) echo "model-serve.sh: unknown model kind $kind" >&2; exit 2 ;;
esac
body() {
  cd "$BOX_TREE"
  nix develop -c "$BUILD/unholy" serve --model "$model" --profile "$profile" --port 8300 \
    --served-model-name "$name" "${SERVE_FLAGS[@]}" "${extra[@]}" >"$d/serve.log" 2>&1 &
  local pid=$!
  until grep -q "^serving $name" "$d/serve.log"; do
    kill -0 $pid 2>/dev/null || { tail -20 "$d/serve.log"; return 1; }
    sleep 2
  done
  local rc=0
  URL=http://localhost:8300 RUN=$d MODEL_NAME=$name "$@" || rc=$?
  stop_server "unholy serve --model $model" || rc=1
  wait $pid 2>/dev/null || true
  return $rc
}
with_gpus body "$@"
echo "server log: $d/serve.log"

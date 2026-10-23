#!/usr/bin/env bash
# Concurrency sweep: the perf tree's server (production flags plus any given here) and check-source's
# gsm8k-chat at 1 / 2 / 4 / 8 concurrent requests. Outputs and the server log in the run directory.
#   tools/box/sweep.sh LABEL [unholy serve flags]
# Env: LIMIT (questions per level, default 200), LEVEL (default high), SEED (default 1234),
#      CONCS (default "1 2 4 8"), PORT (default 8300).
set -euo pipefail
. "$(dirname "$0")/env.sh"
label=${1:?label}; shift
d=$(run_dir "sweep-$label")
port=${PORT:-8300}

sweep() {
  # its own process group, so stopping it also stops what nix develop started
  setsid bash -c 'cd "$1" && shift && exec nix develop -c "$@"' _ "$BOX_TREE" "$BUILD/unholy" serve \
    --model "$MODEL" --profile "$PROFILE" --host 127.0.0.1 --port "$port" --served-model-name dsv4-flash \
    "${SERVE_FLAGS[@]}" "$@" > "$d/server.log" 2>&1 &
  local srv=$!
  trap 'kill -- -'"$srv"' 2>/dev/null; wait '"$srv"' 2>/dev/null' RETURN
  until curl -fs "http://127.0.0.1:$port/v1/models" >/dev/null; do
    kill -0 $srv 2>/dev/null || { echo "server exited; see $d/server.log" >&2; return 1; }
    sleep 5
  done
  for c in ${CONCS:-1 2 4 8}; do
    nix run "$CHECK_SOURCE" -- gsm8k-chat --base-url "http://127.0.0.1:$port" --model dsv4-flash \
      --level "${LEVEL:-high}" --seed "${SEED:-1234}" --limit "${LIMIT:-200}" --concurrency "$c" \
      --out "$d/gsm8k-c$c.json" 2>&1 | tail -3
  done
}
with_gpus sweep "$@" 2>&1 | tee "$d/sweep.txt"

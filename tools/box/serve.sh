#!/usr/bin/env bash
# The perf tree's server with the production flags, in the foreground on $PORT (default 8300). The llm-mode
# services are stopped first and the mode is restored when the server exits. Extra flags are appended.
#   tools/box/serve.sh [unholy serve flags]
set -euo pipefail
. "$(dirname "$0")/env.sh"
with_gpus in_nix "$BUILD/unholy" serve --model "$MODEL" --profile "$PROFILE" --host 0.0.0.0 \
  --port "${PORT:-8300}" --served-model-name dsv4-flash "${SERVE_FLAGS[@]}" "$@"

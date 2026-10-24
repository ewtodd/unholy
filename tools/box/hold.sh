#!/usr/bin/env bash
# Run several GPU scripts under one service stop: the services stay down between them and the serving mode
# comes back once, at the end (also on failure or Ctrl-C).
#   tools/box/hold.sh bash -c 'tools/box/validate.sh && tools/box/curve.sh x && tools/box/eb.sh x 8'
set -euo pipefail
. "$(dirname "$0")/env.sh"
with_gpus env BOX_GPUS_HELD=1 "$@"

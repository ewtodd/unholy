#!/usr/bin/env bash
# Build gufo's routed-expert benchmark: tools/bench/gufo_routed.hip + gufo's MIT FP16 kernel, compiled from a checkout
# (GUFO_SRC, pinned rev; not vendored) into $BUILD/gufo-routed-bench. Builds only; runs nothing on the GPU.
#   tools/box/gufo-routed.sh
set -euo pipefail
. "$(dirname "$0")/env.sh"
GUFO_SRC=${GUFO_SRC:-/work/src/gufo}
rev=01fb6f4e39464003efffd97d79607005146280f0
if [ ! -d "$GUFO_SRC/.git" ]; then
  mkdir -p "$(dirname "$GUFO_SRC")"
  git clone https://github.com/franzmoca/gufo "$GUFO_SRC"
  git -C "$GUFO_SRC" checkout --detach "$rev"
fi
have=$(git -C "$GUFO_SRC" rev-parse HEAD)
[ "$have" = "$rev" ] || { echo "gufo-routed: $GUFO_SRC is at $have, need $rev" >&2; exit 1; }
kdir=$GUFO_SRC/src/models/deepseek_v41_flash/kernels/rocm
mkdir -p "$BUILD"
# shellcheck disable=SC2016  # the compile line expands inside the dev shell
in_nix bash -c '"$CMAKE_HIP_COMPILER" -D__HIP_PLATFORM_AMD__=1 -O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx1151 \
  -I"$1" -x hip tools/bench/gufo_routed.hip -x hip "$1/ds41_routed_f16.hip.cpp" \
  --hip-link -L"$ROCM_PATH/lib" -lamdhip64 -Wl,-rpath,"$ROCM_PATH/lib" -o "$2"' _ "$kdir" "$BUILD/gufo-routed-bench"
echo "built $BUILD/gufo-routed-bench"

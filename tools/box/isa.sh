#!/usr/bin/env bash
# Device assembly of one source file for one GPU, with the build's flags (for counting a kernel's instructions).
#   tools/box/isa.sh src/kernels/exl3_moe.hip gfx1201 OUT.s
set -euo pipefail
. "$(dirname "$0")/env.sh"
src=${1:?source file}; arch=${2:?gfx1201 or gfx1151}; out=${3:?output .s}
in_nix bash -c '"$CMAKE_HIP_COMPILER" -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -Isrc -Ithird_party \
  -O3 -DNDEBUG -std=gnu++20 --offload-arch="$1" --cuda-device-only -S -x hip "$2" -o "$3"' _ "$arch" "$src" "$(realpath -m "$out")"

#!/usr/bin/env bash
# Copy this checkout to son-of-anton:/work/unholy-perf and build it there. Run on the dev machine.
#   tools/box/sync.sh            sync + build
#   tools/box/sync.sh --no-build
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
host=${BOX_HOST:-son-of-anton}
tree=${BOX_TREE:-/work/unholy-perf}

rsync -a --delete --exclude /build/ --exclude /.git/ --exclude /result "$here/" "$host:$tree/"
rev=$(git -C "$here" rev-parse --short HEAD)
git -C "$here" diff --quiet HEAD || rev="$rev+dirty"
ssh "$host" "echo $rev > $tree/.synced-rev"
echo "synced $rev to $host:$tree"
[ "${1:-}" = --no-build ] && exit 0

ssh "$host" "cd $tree && nix develop -c bash -c '
  test -f build/build.ninja || cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=\$CMAKE_HIP_COMPILER \
    -DCMAKE_HIP_COMPILER=\$CMAKE_HIP_COMPILER -DCMAKE_HIP_COMPILER_ROCM_ROOT=\$ROCM_PATH -DCMAKE_PREFIX_PATH=\$ROCM_PATH
  ninja -C build'"

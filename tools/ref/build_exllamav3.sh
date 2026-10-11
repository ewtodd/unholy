#!/usr/bin/env bash
# Builds exllamav3 (the EXL3 reference: dequantization and a full DSV4 forward) in place against the
# vLLM stack's ROCm torch on son-of-anton, for the R9700s (that torch is compiled for gfx1201 only).
# Run inside `nix develop` (for the SDK clang wrapper and a host C++ compiler).
#   tools/ref/build_exllamav3.sh [dest]   then   PYTHONPATH=<dest> $REF_PY ...
set -euo pipefail
DEST=${1:-/scratch/unholy-ref/exllamav3}
REV=${EXLLAMAV3_REV:-3a513ac}
REF_PY=${REF_PY:-/nix/store/vw9rx7ibj1q7m5asl1k97gbnk1gm0a5s-python3-3.14.7-env/bin/python3}
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/turboderp-org/exllamav3 "$DEST"
fi
git -C "$DEST" fetch -q origin && git -C "$DEST" checkout -q "$REV"
# marisa_trie is only used by the n-gram/grammar helpers; the reference never reaches them.
mkdir -p "$DEST/.stubs"
[ -f "$DEST/.stubs/marisa_trie.py" ] || echo "class Trie: pass" > "$DEST/.stubs/marisa_trie.py"
export ROCM_HOME=$ROCM_PATH HIP_PATH=$ROCM_PATH
export HIP_CLANG_PATH=$(dirname "$CMAKE_HIP_COMPILER")/../llvm/bin
export PYTORCH_ROCM_ARCH=gfx1201
export MAX_JOBS=${MAX_JOBS:-24}
# The nix torch does not ship pybind11 in torch/include; use the env's own (the version torch was built with).
export CPATH=$("$REF_PY" -c "import pybind11; print(pybind11.get_include())")${CPATH:+:$CPATH}
cd "$DEST"
PATH=$(dirname "$REF_PY"):$PATH "$REF_PY" setup.py build_ext --inplace 2>&1
echo "EXLLAMAV3_BUILD_DONE"

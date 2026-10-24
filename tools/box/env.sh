# shellcheck shell=bash disable=SC2034
# Settings and helpers shared by the son-of-anton dev scripts (sourced, not run).
# Everything they write lives under /work (the SATA dev array); /scratch holds model weights only.

: "${BOX_TREE:=/work/unholy-perf}"       # the checkout tools/box/sync.sh copies and builds
: "${BOX_DATA:=/work/data}"              # inputs made once: token ids, oracle, request sets, baselines
: "${BOX_RUNS:=/work/runs}"              # one directory per run: <date>-<label>
: "${MODEL:=/scratch/dsv4-ve}"
: "${PROFILE:=$BOX_TREE/data/expert_profile_unholy_mixed.txt}"
: "${IDS:=$BOX_DATA/ids_code32768.bin}"
: "${ORACLE:=$BOX_DATA/oracle/code1024}"
: "${REF_PY:=/work/env/py}"              # python with ROCm torch (tools/box/setup.sh makes it)
: "${CHECK_SOURCE:=github:ewtodd/check-source/38c265876c47d25c288851c6bc6838caff81a118}"
# GLM-5.3-Flash (docs/PLAN-glm-dsv41.md, G2): FP8 release, its token ids, the transformers oracle and its venv
: "${GLM_MODEL:=/scratch/models/glm53-flash-fp8}"
: "${GLM_IDS:=$BOX_DATA/glm/ids_code32768.bin}"
: "${GLM_ORACLE:=$BOX_DATA/glm/oracle/code1024}"
: "${GLM_PY:=/work/env/glm/bin/python}"
: "${TRANSFORMERS_REV:=50e35cc0ca18ec19b80653690de553e6decf74f5}"   # transformers main with glm5_next
BUILD=$BOX_TREE/build

# The production server's flags (the NixOS module's, without host / port / name).
SERVE_FLAGS=(--kv-tokens 1572864 --max-model-len 786432 --max-num-seqs 8 --max-batch-tokens 4096
             --prefix-snapshots 64 --spec-drafts 5)

# run_dir LABEL: a fresh directory for one run's outputs.
run_dir() {
  local d="$BOX_RUNS/$(date +%F)-$1" n=1
  while [ -e "$d" ]; do d="$BOX_RUNS/$(date +%F)-$1.$((n++))"; done
  mkdir -p "$d"
  echo "$d"
}

# in_nix CMD...: run inside the tree's dev shell (the binaries need the TheRock ROCm environment).
in_nix() { (cd "$BOX_TREE" && exec nix develop -c "$@"); }

# with_gpus CMD...: stop every llm-mode service, run CMD, then restore the mode that was serving, also
# when CMD fails or is interrupted. son-of-anton may manage these units without sudo (polkit).
# Inside tools/box/hold.sh the services are already down and stay down: CMD just runs.
with_gpus() {
  if [ -n "${BOX_GPUS_HELD:-}" ]; then "$@"; return; fi
  local mode rc=0
  mode=$(cat /var/lib/llm-mode/mode 2>/dev/null || echo ds4)
  trap 'llm-mode "'"$mode"'" >/dev/null' EXIT
  trap 'exit 130' INT TERM
  systemctl stop unholy.service radiance.service gufo-strix.service
  "$@" || rc=$?
  llm-mode "$mode" >/dev/null
  trap - EXIT INT TERM
  return $rc
}

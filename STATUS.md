# STATUS

Live state. Design and targets: `DESIGN.md`; history and evidence: `DECISIONS.md`.

## Where things are

- Source of truth: `github.com/ewtodd/unholy` (private), working tree on e-desktop
  `~/Software/unholy`; build/run tree `son-of-anton:/scratch/unholy` (rsync, no `.git`).
- Weights: native `/scratch/dsv4-ve` (48 shards, 167.8 GB); EXL3 3.04 bpw
  `/scratch/models/dsv4-ve-exl3-3.04bpw` (16 shards, 118.4 GB).
- Toolchain: `github:ewtodd/therock-sdk-nix` (ROCm 10.1.0, gfx120X-all dist; fat binaries for
  gfx1201 + gfx1151).
- son-of-anton GPUs: `vllm.service` and `gufo-strix.service` are stopped; the owner has freed
  all three devices for this work.

## Build / run (on son-of-anton)

```
cd /scratch/unholy
nix develop -c bash -c 'cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=$CMAKE_HIP_COMPILER \
  -DCMAKE_HIP_COMPILER=$CMAKE_HIP_COMPILER -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH \
  -DCMAKE_PREFIX_PATH=$ROCM_PATH && ninja -C build'
nix develop -c ./build/unholy-probe --json probe.json
```

## Milestone 1 (foundation)

- [x] flake + CMake (TheRock 10.1 SDK, fat gfx1201/gfx1151)
- [x] `unholy-probe`: PCIe paths, verified transfers, signalling latency (DECISIONS 2026-10-11)
- [x] checkpoint reader + CPU reference decoders: native (MXFP4, fp8 block-scaled) and EXL3 (bit-exact vs exllamav3)
- [ ] device weight loader (placement-driven upload)
- [ ] memory planner dry-run against the probe JSON (4 x 512k slots, hot/cold expert split)

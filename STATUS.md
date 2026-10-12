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
- [x] device weight loader (O_DIRECT coalesced uploads into per-device arenas; full resident load ~21 s)
- [ ] memory planner dry-run against the probe JSON (4 x 512k slots, hot/cold expert split)

## Milestone 2 (correct forward)

- [x] oracle: DeepSeek's model.py layer-streamed on one R9700 (`tools/ref/oracle`), 2606/223/18/11/4054
- [x] reference kernels (`src/kernels/ref_*.hip`) and the reference layer forward (`src/model/dsv4.hip`):
      reference rounding points, chunked prefill, window ring + compressor rings + paged compressed rows
- [x] per-layer comparison against the oracle (`unholy-layer-test`) and intermediate dumps
      (`UNHOLY_DUMP`, `tools/ref/compare_dump.py`)
- [x] full-model logits vs oracle (`unholy-ref-run`): mean KL 9.6e-3, top-1 95.7% vs the oracle's own
      noise floor 9.9e-3 / 96.2% (DECISIONS 2026-10-11) — correct
- [ ] DSpark (the checkpoint's `mtp.*` block drafter) on the reference path

## Milestone 3 (serving)

- [x] tokenizer (`src/text/tokenizer.cpp`, PCRE2): identical to HF `tokenizers` on 91k adversarial +
      real strings except code points assigned after HF's Oniguruma tables; decode identical;
      5.3M tok/s (`tools/ref/tok_compare.py`)
- [x] chat format + streaming output parser (`src/text/chat.cpp`): byte-identical to
      `encoding_dsv4.py` (golden 4/4, 20k random conversations), request path identical to vLLM's
      `apply_chat_template` (text and ids), parser identical to the reference at any chunking
      (`tools/ref/chat_compare.py`)
- [ ] scheduler + paged cache manager (continuous batching, chunked prefill, slots)
- [ ] OpenAI-compatible HTTP server (SSE), `unholy` CLI, NixOS module

### Reference path vs oracle, per layer (1024 prefill + 8 decode, code prompt)

| layer | type | prefill rel L2 | decode rel L2 | bit-identical |
|---|---|---|---|---|
| 0 | window, hash-routed | 1.1e-2 | 1.3e-2 | 73% |
| 2 | CSA, hash-routed | 4.5e-3 | 3.1e-3 | 59% |
| 3 | HCA | 7.7e-3 | 5.2e-3 | 44% |

Bisected on layer 0 (`compare_dump.py`): projections 99.9% bit-identical; the attention kernel on the
oracle's exact inputs 98.5% (rel 4e-4); the wo_a einsum turns the 1.5% one-ulp differences into 12%
(K = 4096 sums); every fp8 activation quantization then amplifies ~3x (an e4m3 step is 12.5%, so a
0.4% input change flips some roundings): wo_b 3e-3, MoE input 3.8e-3, MoE output 1.2e-2. Inherent to
re-implementing a QAT'd fp8-activation model with a different fp32 summation order; judged end to end.

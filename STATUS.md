# STATUS

Live state. Design and targets: `DESIGN.md`; history and evidence: `DECISIONS.md`.

## Where things are

- Source of truth: `github.com/ewtodd/unholy`; the build/run tree on son-of-anton is
  `/scratch/unholy` (rsync, no `.git`).
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
nix develop -c ./build/unholy serve --model /scratch/dsv4-ve --exl3 /scratch/models/dsv4-ve-exl3-3.04bpw \
  --profile data/expert_profile_dsv4engine_code32k.txt --max-slots 4 --max-seq 32768 --max-batch-tokens 1024 --port 8100
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
- [x] DSpark (the checkpoint's `mtp.*` block drafter): matches the reference; speculative decoding
      with exact acceptance, ring backups/restores, --spec-drafts (DECISIONS 2026-10-12)

## Milestone 3 (serving)

- [x] tokenizer (`src/text/tokenizer.cpp`, PCRE2): identical to HF `tokenizers` on 91k adversarial +
      real strings except code points assigned after HF's Oniguruma tables; decode identical;
      5.3M tok/s (`tools/ref/tok_compare.py`)
- [x] chat format + streaming output parser (`src/text/chat.cpp`): byte-identical to
      `encoding_dsv4.py` (golden 4/4, 20k random conversations), request path identical to vLLM's
      `apply_chat_template` (text and ids), parser identical to the reference at any chunking
      (`tools/ref/chat_compare.py`)
- [x] scheduler + paged cache manager (`src/engine/engine.cpp`): continuous batching, decode-first
      steps with chunked prefill in the remaining budget, shared page pool with recompute preemption,
      slots; verified against a fake paged model under contention (`unholy-sched-test`)
- [x] OpenAI-compatible server (`src/server/openai.cpp`, cpp-httplib): chat + completions with SSE,
      tool calls, reasoning (`reasoning` and `reasoning_content`), stop strings, logprobs, usage,
      /tokenize, /detokenize, /metrics; `unholy serve` (`tools/unholy/main.cpp`)
- [x] NixOS module / service (/etc/nixos modules/services/unholy: native experts, 2M-token pool, 1M per
      request; polkit lets son-of-anton stop/start it for development)
- [x] prefix caching (page hashes + slot-state snapshots; logprobs identical to uncached), vLLM-style
      KV flags, cached_tokens in usage
- [x] GPU fault: c7 under the 230 W power cap (DECISIONS 2026-10-12); clean at 300 W
- [x] vision: CPU preprocessing bit-exact with the reference, ViT + Aligner on the APU at the reference's
      own noise floor, image tokens end to end (DECISIONS 2026-10-12)
- [x] logging: a line per finished request (queue, ttft, prefill / decode tok/s, draft acceptance) and a
      periodic engine line (`--log-interval`, default 10 s: throughput, queue, KV use, prefix hit rate,
      draft acceptance by position); request log checked against deliveries in `unholy-sched-test`
- [x] PLX switch (both cards on one Gen4 x4 uplink): compact card <-> APU expert transport (fp8 inputs per
      token, bit-identical; outputs combined per token on the APU): prefill 1062 tok/s average to 512k
      (971 before the switch), card <-> card 26 GB/s (DECISIONS 2026-10-12)
- [x] decode latency (DECISIONS 2026-10-14/15): single-stream 56.7 -> 34.2 ms/token at 4k, 87.6 -> 44.8 ms at
      512k (W8A8 GEMV fp8 decode, single-workgroup kernels spread, attention staging bank conflict and split
      across workgroups, top-k on several workgroups for long rows, fewer launches); bit-identical logits
- [ ] disk tier for the prefix cache
- [x] decode scaling: one attention / top-k launch for all chunks, GEMVs up to 16 rows and MoE GEMVs up to 64
      tokens (opt-in: they change those rows' values), two decode micro-batches across the cards
      (`--decode-split`): batch 8 step 168 -> 87 ms; GSM8K sweep 22.2 / 36.5 / 45.5 / 56.3 -> 29.4 / 54.0 / 56.0 /
      61.4 tok/s at 1 / 2 / 4 / 8 by default, 29.9 / 55.4 / 68.6 / 82.5 with the opt-ins (DECISIONS 2026-10-15)
- [ ] multi-stream decode and prefill are bound by the cold experts' bytes on the APU (65-82% busy): more
      routing on the cards (scratch went 3.21 -> 2.24 GB per card; KV pool size is the other lever) or a
      faster APU grouped GEMM (20-25 TFLOPS of ~59 at 84-280 rows per expert)
- [ ] verify steps of more than 8 rows (16 with UNHOLY_GEMV_ROWS=16) use the WMMA GEMMs: their rows are not
      bit-identical to single-stream ones, and those GEMMs are slow at small M (DECISIONS 2026-10-14)
- [x] two steps in flight (card pipelining): 16k prefill 1087 -> 1680 tok/s
- [x] routing profile from our own traffic (`data/expert_profile_unholy_mixed.txt`, default for runs)

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

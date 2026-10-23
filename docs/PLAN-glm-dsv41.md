# Plan: GLM-5.3-Flash and DeepSeek-V4.1-Flash-1017 as second modes

Date: 2026-10-23. Approved by the owner in conversation the same day; the service windows are still to be set.

Goal: serve GLM-5.3-Flash and an EXL3 3.0 quant of DeepSeek-V4.1-Flash-1017 on son-of-anton as extra
`llm-mode`s next to the DSV4-Vision-Exp served today, and compare all three (plus the qwen mode, for interest)
on performance and intelligence. The point of local serving is privacy, not cost. DSV4-VE stays the default
mode throughout. The hardware does not change. Background numbers: `docs/glm53-flash-quant.md` (GLM) and
the DSV4.1 streaming estimates in "D3" below.

GLM goes first: it fits in memory, turboderp's EXL3 quants exist, and it needs no SSD tier; the hard part is new
kernels (KDA, DSA, a new router) and making the engine model-agnostic. V4.1 reuses more of the DSV4 code but needs
a 543.6B-parameter quantization run of our own and an SSD streaming tier, and its results will be bound by the SSD.

## Ground rules

- **Privacy:** hosted APIs only ever see public benchmark items (gsm8k, math500, AIME, the synthetic longctx
  documents, the public coding bench). Knoll stays on local engines until it is public.
- **DSV4 regression gate:** every engine change keeps DSV4-VE's output bit-identical to the M0 baselines
  (resident test + engine-batch, same expert placement); checked at every milestone.
- **No clutter on the box:** build trees, profiles, oracle dumps, eval outputs and logs live under `/work`;
  scripts worth keeping are committed to this repo (`tools/box/`); `/scratch` holds model weights only.
- **Service windows:** GPU work that needs the serving mode stopped (speed curves, oracle runs on a card,
  resident tests, quantization, the qwen baselines) may run at any time for now (owner, 2026-10-23), until
  the owner announces constraints; the previous `llm-mode` is always restored afterwards.
- **Each milestone ends** with a commit, a DECISIONS entry with its numbers, and a short report. Decision
  points for the owner are marked **Decision**.
- **Implementation:** self-contained coding goes to the local model (delegate-local) while the service is up;
  GPU work happens in the windows.

## Hosted references

| model | endpoint | notes |
|---|---|---|
| GLM-5.3-Flash | OpenRouter, pinned to Z.AI (fp8, the release precision) and Near AI (fp8, returns logprobs) | `provider.only` + `allow_fallbacks: false` + a `quantizations` filter; the serving provider is logged per response and items served elsewhere are dropped |
| DeepSeek-V4.1-Flash-1017 | DeepSeek API, model `deepseek-flash` | DeepSeek's docs: "now served by DeepSeek-V4.1-Flash-1017". Never use the legacy names `deepseek-v4-flash` / `deepseek-v4-flash-vision-exp`: they are answered by V4.1-1017 without an error. Log the returned `model` and the date per run; recheck the docs before runs (the alias moves) |
| DSV4-VE (calibration) | OpenRouter, GMICloud or DeepInfra (fp8) | the same evals on hosted and local DSV4-VE; if they match within noise, the hosted rows for the other models can be trusted |

Once the oracles exist (G2, D4), the logprob-returning endpoints are checked against them on fixed prompts.
Hosted runs are cheap, so they get more seeds (4-5 on AIME). The DeepSeek key stays on the local machine; hosted
evals run from there (the local engines listen on all interfaces).

## M0: Groundwork (3-4 days)

1. **SATA drives** (done 2026-10-23). Six Micron 1100 256 GB on the ASM1166 (PCIe Gen3 x2):
   - `/stream`: 3 drives, RAID 0, 1 MiB chunks, 703 GB: V4.1's Engram tables (203 GB) and the SATA share of
     its streamed experts.
   - `/work`: the other 3, RAID 0, 703 GB: everything that is not model weights or the stream pool.
   - Declared in `/etc/nixos` (`boot.swraid` + `fileSystems` with `nofail`). Both share the controller, so
     `/work` stays quiet during streaming benchmarks.
   - The ASM1166 trained its link at Gen1 (2.5 GT/s, `/stream` 0.44 GB/s) with firmware 221118-0048-00;
     ASMedia's 241025-0000-05 (Radxa's build, flashed with `116xfwdl` after dumping the original through a
     reader written for it) trains at Gen3. Measured with fio (direct I/O): `/stream` 1.62 GB/s (8 MB
     sequential and 16 MB random), 177k 4 KB random reads/s; NVMe 6.2 GB/s (16 MB random); both at once
     7.78 GB/s.
2. **check-source additions** (deferred 2026-10-23 while the owner works on check-source; resume once Knoll
   lands; the hosted references and G1 / D1 gates wait for this): API key from an environment variable,
   `--extra-body` (OpenRouter provider pinning and `reasoning`, DeepSeek `thinking` + `reasoning_effort`),
   the serving provider and returned `model` logged per item, and a **coding bench**: code generation with
   hidden tests (LiveCodeBench's recent problems, to limit contamination), executed in a bubblewrap sandbox
   without network. No tool-call bench for now.
3. **Dev setup, committed this time:** `tools/box/` (validate, bench, curve, serve-load, sweep, engine-batch
   helpers); on `/work`: the perf tree, the DSV4 oracle on the 1024-token code prompt, token ids, baselines.
4. **Comparison protocol** (`docs/compare-protocol.md`):
   - Intelligence: Knoll (once active), the coding bench, gsm8k-chat (200), math500 (200), AIME 24/25/26,
     longctx at 32k / 128k / 512k (256k cap for the qwen models). Fixed seeds, reasoning levels mapped between
     models, paired statistics.
   - Performance, through the API so every engine is measured the same way: prefill rate at depth (time to
     first token with `longctx --stream`), decode tok/s from streamed output, total throughput at 1/2/4/8
     streams, memory headroom. Engines we build also get the internal curves (`unholy-curve`).
5. **Baselines:**
   - DSV4-VE local (evals against the live service; speed runs in a window);
   - DSV4-VE hosted (calibration row);
   - Qwen3.8-27B (radiance, fp8, both R9700s, up to 352k) and Qwen3.8-Flash-Next (gufo, Q4 + MTP, Strix Halo,
     up to 262k) under `llm-mode qwen`, one at a time with the other idle; DSV4 is offline for these runs, so
     they take a window.

## GLM-5.3-Flash

**G1: Gate (1-2 days, mostly runs).** Full-precision GLM on the M0 evals through the pinned OpenRouter endpoints.
If it beats DSV4-VE, also run Maya-L (Project Maya's ~3.9 bpw quant, built under `/work`, in a window) on the same
evals as a local-quant proxy before the port. **Decision:** go / no-go.

**G2: Weights and oracle (2 days).** FP8 checkpoint (328 GB) and EXL3 3.05 + 4.05 (290 GB) to `/scratch/models`.
Oracle: HF transformers `glm5_next` on the FP8 checkpoint, one layer at a time on one R9700, from a pinned
environment on `/work`: teacher-forced logits on the code prompt and mixed prompts, per-layer dumps, the noise
floor (the oracle against itself with reordered arithmetic), and a router histogram (the first GLM routing
profile). Exit: the oracle reproduces HF generation; noise floor measured.

**G3: Model-agnostic engine (2-3 days).** A model-family interface for the loader and layer types, the planner
sizes, the cache manager (per-sequence recurrent state, double-buffered for speculative decoding; latent KV pages;
pooled indexer rows with a per-sequence tail carry), and the tokenizer / chat template / output parser. Exit:
DSV4 bit-identical to the M0 baselines, no speed regression.

**G4: Reference forward (3-4 days).** Reference kernels for KDA (conv, lower-bounded gate, chunked + recurrent
delta rule, gated norm), DSA (pooled indexer, top-2048 + tail, absorbed NoPE MLA), the sigmoid router (bias,
x2.5), dense layers 0-2, mHC, EXL3 experts (288, top-8) and both dense formats. Per-layer vs the oracle, then
full-model KL / top-1. Exit: native path at the noise floor; KL table for builds A-D
(`docs/glm53-flash-quant.md`). **Decision:** 3.05 vs 4.05 experts, EXL3 vs fp8 dense.

**G5: Fast path (4-6 days).** WMMA kernels for KDA prefill and decode, the DSA indexer and attention, and the
chosen dense format (EXL3 dense GEMV, or fp8 KDA weights); placement from the GLM profile; two steps in flight;
MTP drafting with KDA state rollback. Exit: KL unchanged from the reference path; measured against >= 500 tok/s
prefill and >= 20 tok/s decode at 512k (reported honestly if missed).

**G6: Serving (2 days).** Scheduler and prefix-cache policy for the 152 MB KDA snapshots, the GLM chat template
(`reasoning_effort`, tool calls), the NixOS module option and `llm-mode glm`. Exit: the mode serves;
concurrency checks pass.

**G7: Comparison (1-2 days of runs).** Report in `docs/`, DECISIONS entry. **Decision:** keep the GLM mode or
not. GLM vision is deferred.

## DeepSeek-V4.1-Flash-1017, EXL3 3.0

**D1: Gate and weights (1-2 days).** Full-precision V4.1 through the DeepSeek API on the M0 evals. Fetch the
remaining 44 shards (~304 GB) to `/scratch/models`; reuse the 4 in `~/engram/dsv41-flash-1017` (incl. the 203 GB
Engram tables, which move to `/stream`, with the owner's OK since engram-probe reads them). Confirm exllamav3
supports `deepseek_v41`. **Decision:** go / no-go.

**D2: EXL3 3.0 quantization (2-4 days, mostly compute, one long window).** Experts at 3.0 bpw from the native
MXFP4, dense weights stay native fp8; working files on `/work`. Weight-space error vs the native weights.
Fallback if it fails: LibertAIDAI's 3.5 bpw build (80.7 GB on the SSD instead of 44.9).

**D3: SSD tier, proven on DSV4-VE first (3-4 days).** Experts hot on the cards, warm on the APU, cold on the SSD;
the cold set split between the NVMe (6.2 GB/s) and `/stream` (1.6 GB/s) by bandwidth, read with io_uring
direct I/O into an APU staging buffer (the service needs `LimitMEMLOCK` raised: io_uring fails under the
default 8 MiB); planner support; an Engram store that prefetches rows from the token ids. Tested by forcing
some of DSV4's experts onto the SSD: output must stay bit-identical; cost measured against GB on the SSD,
which checks these estimates (4x512k, 45 ms compute per token, NVMe + `/stream` 7.78 GB/s measured):

| V4.1 build | experts | on the SSD | prefill, 4k-token steps | decode, 1 stream at 512k | all streams |
|---|---:|---:|---:|---:|---:|
| EXL3 3.0 | 204.7 GB | 44.9 GB | ~710 tok/s | 15-19 tok/s | 49-122 tok/s |
| MXFP4 (lossless, not planned) | 288.8 GB | 135.7 GB | ~235 tok/s | 7-10 tok/s | 11-17 tok/s |

**D4: V4.1 model (4-5 days).** Changes from DSV4: hidden 5120, 384 experts, top-6, expert width 2304; KV and
indexer shared across layers; compression at ratios 2 and 1; candidate pre-filtering (layer 20, 2048 blocks of 8);
Engram (hash, 48-row lookup, wkv, per-stream gate); the DSpark drafter with 128 experts, top-3; fp8 with 32x32
scale blocks (our fp8 kernels assume 128x128); the V4.1 chat encoding. Oracle: DeepSeek's `inference/model.py`
from the checkpoint, one layer at a time, Engram tables memory-mapped from `/stream`. Exit: native path at the
noise floor; EXL3 3.0 KL.

**D5: Fast path and serving (3-5 days).** New shapes on the fast path; the SSD tier in the scheduler (large
prefill passes, decode priority); DSpark measured on and off (expected to hurt when SSD-bound); `llm-mode ds41`.

**D6: Three-way comparison.** Report and DECISIONS entry. **Decision:** keep the V4.1 mode or not.

## Effort

Days of active work, judged against DSV4's pace (first commit 2026-10-11 to the tuned served engine on 10-15):

| | days |
|---|---:|
| M0 | 3-4 |
| GLM, G1-G7 | 15-21 |
| V4.1, D1-D6 | 14-22 |

## Decisions taken (2026-10-23)

- SATA: `/stream` = 3 drives RAID 0, `/work` = 3 drives RAID 0.
- Maya-L proxy: run in G1, only after the hosted gate passes (Claude's call; the owner left it open).
- Coding bench: add one (LiveCodeBench recent problems, sandboxed); no tool-call bench for now (Claude's call).
- check-source additions: Claude, on a branch.
- Service downtime: fixed windows (times to be set by the owner).
- Hosted references: OpenRouter for GLM and the DSV4-VE calibration row, the DeepSeek API for V4.1.

# unholy — design

A vertical serving engine for **DeepSeek-V4-Flash** on one machine, `son-of-anton`:
two Radeon AI PRO R9700 (gfx1201, 32 GB each) and the Strix Halo iGPU (gfx1151, ~124 GB
GTT carved from the 128 GB LPDDR5X the host also runs from). C++20 + HIP, no Python at
runtime, packaged with nix on the TheRock ROCm 10 SDK.

`STATUS.md` holds the live state and measured numbers; `DECISIONS.md` holds dated decisions
with their evidence. This file is the target and the shape of the system.

## Requirements (hard, from the owner, 2026-10-11)

| | requirement |
|---|---|
| quant | no lower than **3 bits per weight** (EXL3 first; own formats allowed) |
| prefill | **>= 500 tok/s at 512k depth** (the marginal rate, not an average), as high as possible |
| context | **multiple slots of 512k** each — planned for **4** |
| decode | **>= 20 tok/s per stream at long context**, speculative decoding included |
| concurrency | **vLLM/sglang class**: continuous batching, paged state, prefix caching, chunked prefill, preemption |
| toolchain | the existing **TheRock ROCm 10** nix packages (`github:ewtodd/therock-sdk-nix`) |
| packaging | **nix** flake + NixOS module |
| topology | must keep working when the R9700s move behind a **PLX switch** (x16 card<->card, one **shared** x4 uplink to the Strix Halo) |

Scope order: text + speculative decoding first; vision (the 0.47B ViT + aligner) after the
text engine meets its targets, with hooks designed in from the start (see "Vision").

## Model

`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (284B total, ~13B active), local at
`/scratch/dsv4-ve` (native: MXFP4 routed experts, fp8 e4m3 dense with ue8m0 128x128 block
scales, bf16 embed/head/compressors). EXL3 3.04 bpw (turboderp, exllamav3 1.4.8, `mul1`
codebook: experts and MTP at 3 bits, attention/head/vision at 5) at
`/scratch/models/dsv4-ve-exl3-3.04bpw`.

DeepSeek-V4.1-Flash (2026-10-01) is ~550B (384 experts, hidden 5120): ~205 GB at 3 bpw,
which does not fit this machine. Out of scope.

Shape summary (full map in `docs/ARCHITECTURE.md` once ported from dsv4-engine): 43 layers,
hidden 4096, 4 hyper-connection streams (16384-wide residual), 64 query heads x 512, one
512-wide KV head (MQA), q/o through LoRA ranks 1024 (o in 8 groups), rope dim 64 (YaRN x16
from 64k). Layers 0-1 sliding window (128); 21 CSA layers (4:1 compressed KV + Lightning
Indexer, 64 heads x 128, top-512); 20 HCA layers (128:1 compressed KV, dense over it). 256
routed experts (6 active, inter 2048) + 1 shared, sqrtsoftplus gating, layers 0-2
hash-routed by token id. 3 MTP layers.

## Budget (why the targets are reachable, and what they cost)

Memory, 4 slots x 512k:

| item | size | where |
|---|---|---|
| routed experts, EXL3 3.04 bpw | ~105 GB (+2.4 GB per MTP layer) | hot on cards, cold on APU |
| dense (attention fp8, shared experts, compressors, indexer, HC, router) + head | ~8 GB | cards |
| compressed KV (fp8 nope + bf16 rope + e8m0 scales) + indexer keys (fp4) | ~1.8 GB per 512k slot -> ~7.4 GB (bf16 would be 3.6 GB per slot) | cards |
| sliding-window rings + compressor carries | ~6 MB per sequence | cards |
| activation scratch (8k-token chunks, tiled indexer) | ~3 GB per card | cards |

That leaves ~34 GB of the two cards for **hot experts** (~1/3 of them). dsv4-engine's
calibration put 82 of 256 experts/layer on the cards and caught ~85% of the routing on code
text; usage is workload-dependent, so placement adapts at runtime (Strata's idea). The APU
holds ~71 GB of cold experts, leaving the host ~45 GB.

Compute per prefill token at depth d (FLOPs):

- Lightning Indexer: 21 layers x (d/4) keys x 64 heads x 128 x 2 = 86016 d -> **45 G at 512k**
- HCA attention: 20 x (d/128) x 64 x (512+512) x 2 = 20480 d -> 11 G at 512k
- CSA sparse attention: 21 x 640 x 64 x 2048 = 1.8 G (constant)
- weights: ~2 x 13.4 B active = 27 G

~85 GFLOP/token at 512k: **500 tok/s = ~42 TFLOPS sustained**, more than half of it the
indexer. The indexer must therefore be an fp8 WMMA kernel on gfx1201 (R9700 fp8 dense peak
~380 TFLOPS; the kernels inherited today reach 15-33 TFLOPS in f16/bf16).

Decode bytes per token on the cards: attention fp8 4.6 GB + shared experts 1.1 + compressors
and indexer 0.8 + head 0.5 + card-resident routed experts ~2 GB = **~9 GB**. With the layers
split across the two cards (pipeline), one stream reads them serially: ~18 ms at ~500 GB/s,
i.e. a raw ceiling around 40-50 tok/s before overheads. The overhead that decides decode is
the **43 per-token round trips to the APU** for cold experts: host-orchestrated
(event + memcpy, ~100-200 us each) would cost 4-9 ms/token; device-side signalling must be
the design (see Transport). MTP (3 layers, draft up to 3) on top: DeepSeek reports
~85-90% first-token acceptance.

## Architecture

```
            HTTP / OpenAI (chat, completions, responses, SSE, tools, reasoning)
                                     |
                       Scheduler (continuous batching)
     chunked prefill + decode mix | preemption | spec-decode verify | PP micro-batches
                                     |
        Cache manager: paged compressed KV/indexer pools, per-seq rings+carries,
                   radix prefix cache with state snapshots
                                     |
                 Executor: plan -> per-device programs (HIP graphs)
        card A: layers [0, k)      card B: layers [k, 43)       APU: cold experts
                 \___________________ Transport ___________________/
                  P2P DMA | host-staged | zero-copy | device-side signals
                                     |
                 Topology probe + planner (measured, cached by fingerprint)
```

### Topology probe and planner

At startup the probe enumerates devices and the PCIe path of each (sysfs: every hop's max
link; the **current** link speed is useless at idle — the root ports sit at 2.5 GT/s under
ASPM), detects a shared upstream switch, and **measures**: h2d/d2h per device, card<->card
per method, APU zero-copy reads, small-transfer latency, and the card->APU->card signalling
round trip per mechanism. Every transfer method is **verified** (pattern + readback through a
known-good path) before it is eligible: on this machine `hipMemcpyPeer` has silently
no-op'd (dsv4-engine DECISIONS 2026-10-08), so an unverified method is never used.

The planner turns (model, measured topology, slot/context config) into a plan: layer->card
split, expert placement (hot on the owning card, cold on the APU), KV pool sizes, transport
per device pair, and which intra-layer split (if any) is worth it. The plan is data: moving
the cards behind the PLX switch means a re-probe and a new plan, never a code change. The
plan is cached by topology fingerprint and can be pinned.

Rules that keep the PLX move safe:
- Cross-device traffic is **activations only** (the MoE input/output rows and the 16384-wide
  residual at the pipeline cut). At 1000 tok/s prefill that is ~1 GB/s to the APU, which a
  single shared x4 uplink carries; nothing assumes two independent host links.
- Expert cache refills (APU -> card) are rate-limited background traffic budgeted against the
  measured uplink.
- The pipeline split works on any topology. The x16 card<->card link enables optional
  intra-layer parallelism for decode latency. DSV4 attention is MQA, so head-split TP would
  replicate the KV; the split to use is **context parallel** over KV pages (exact top-k merge
  of per-card indexer results, LSE-merged attention partials).

### Transport

Per device pair, the probe-selected method: direct P2P DMA, host-staged via mapped pinned
memory (dsv4-engine's StageBus), or zero-copy reads for the APU. For decode, the target is
**no host on the critical path**: each device runs its step as a captured graph, and
cross-device dependencies are device-side (`hipStreamWaitValue32`/`WriteValue32` on
fine-grained memory, or a persistent polling kernel on the APU), measured by the probe
before it is relied on.

### Cache manager

DSV4 state per sequence, per layer type:

| layer | paged (grows with context) | per-sequence fixed |
|---|---|---|
| sliding (0, 1) | — | window ring 128 x 512 |
| CSA (21) | compressed KV 1 row / 4 tokens x 512; indexer keys 1 row / 4 tokens x 128 | window ring; compressor + indexer-compressor carry (overlap coff=2) |
| HCA (20) | compressed KV 1 row / 128 tokens x 512 | window ring; compressor carry |

Pages are **256 tokens** (aligned with both ratios: 64 CSA rows, 2 HCA rows). Pools are per
card (each card owns its layers' pages); a block table per sequence maps logical pages.
Prefix caching is a radix tree over 256-token pages. A cached prefix is only resumable with
the fixed state at its boundary, so the tree stores **state snapshots** (window rings +
carries, ~6 MB) at chosen boundaries — end of prompt, turn boundaries, every N pages — the
way sglang/vLLM cache hybrid (Mamba/GDN) models. Snapshots can live in host memory and be
uploaded on a hit.

### Scheduler

vLLM-v1 style: one token budget per step, decode first, prefill chunks fill the rest;
preemption by recompute (or swap of pages to host); two pipeline micro-batches in flight so
both cards stay busy (dsv4-engine's and ds4-hybrid's two-lane result: the single largest
prefill win). Spec decoding is part of the step: drafts for all decoding sequences are
verified in one batched pass, with per-sequence rollback of the fixed state (rings and
carries rewind; paged rows are append-only so truncation is a counter).

### Kernels

Two families, chosen per device at load: gfx1201 (RDNA4: fp8/bf8 WMMA, wave32) and gfx1151
(RDNA3.5: fp16/bf16/iu8 WMMA, no fp8). Every kernel has a reference implementation and a
unit test against it. The hot set:

- dense projections: W8A8 fp8 WMMA on gfx1201 (dsv4-engine `dense_w8a8_kernel`, radiance
  fragment map); W8A16 on gfx1151
- routed experts: EXL3 trellis-decode grouped GEMM/GEMV on both families (exllamav3 ROCm
  path, gfx1151 WMMA tables) and MXFP4 for the mixed-precision study
- Lightning Indexer: fp8 WMMA score + fused streaming top-k
- sparse attention: pre-gathered selected rows + dense tile kernel (ds4-hybrid)
- compressors, HC (Sinkhorn mixing), norms/rope, router: from gufo/ds4.c and dsv4-engine

### Speculative decoding

Proposer interface, chosen by measured acceptance x cost per workload: **DSpark** — the
checkpoint's `mtp.*` layers are a 3-stage DSpark block drafter (block 5, conditioned on target
layers 40-42, Markov head + confidence head for adaptive block length) — and **prompt lookup** (n-gram copy; Strata gates it by
measured payoff). Verification is exact (greedy match or standard rejection sampling), so
output quality is the target model's.

### Vision (after text)

The ViT (32 blocks x 1024, patch 14, ~0.41B) + aligner (3x3 merge -> 4096) is small, and
reference implementations exist (llama.cpp `tools/mtmd/models/deepseek4v.cpp` + `clip.cpp`
preprocessing, exllamav3 `deepseek_v4_vision.py`). Hooks designed in now: a batch row can
carry an embedding override instead of a token id, a per-token modality flag selects the
router's `bias_vl` instead of `bias`, and the prefix-cache key hashes image content.

## Quant plan

Routed experts were trained in MXFP4, so native MXFP4 (4.25 bpw, 147 GB) is lossless and
anything at ~3 bits loses something. Order:

1. Measure: KL divergence and top-1 agreement of EXL3 3.04 against native MXFP4, both run in
   this engine on the same prompts (short context, where MXFP4 still fits).
2. Default: all-EXL3 experts. Dense attention: compare EXL3 5 bpw with native fp8 (fp8 is
   both lossless and the fastest path on gfx1201).
3. Mixed: per-expert format chosen by measured importance (usage x sensitivity): the experts
   that matter most for the owner's workload stay MXFP4, the rest EXL3, within the memory
   budget. Own EXL3 quantization (exllamav3's quantizer has a ROCm path) for other bitrates.

## Validation

- Per-kernel: unit tests against a reference, on both architectures.
- Per-layer: against DeepSeek's reference (`inference/model.py`) and exllamav3's DSV4
  modules on captured activations.
- End-to-end: logits KL / top-1 against exllamav3 running the same EXL3 weights (same
  dequantization), and against gufo/ds4 as an independent implementation. Token-identity
  gates are not used: MoE routing turns any FP reordering into a different token stream
  (dsv4-engine DECISIONS 2026-10-08 fourth).
- Long context: needle retrieval at 128k/256k/512k; concurrency: N streams vs N serial
  runs of the same requests must give the same outputs (greedy, same batch composition).

## Donors (and their licenses; credited in NOTICE)

| source | license | taken |
|---|---|---|
| gufo (gufo-org/gufo), DSV4 part from antirez ds4.c | MIT | DSV4 kernels (HC, compressor, indexer top-k, attention), tokenizer, chat template, HTTP/OpenAI server, continuation cache |
| exllamav3 (turboderp-org) | MIT | EXL3 format and dequant, HIP GEMM/MoE kernels, DSV4 module math |
| dsv4-engine (own) | Apache-2.0 | fp8 W8A8 dense kernel, safetensors loader, StageBus, placement planner, expert profiling |
| llama-halo-hybrid `ds4-hybrid` (llama.cpp fork) | MIT | sparse-FA pre-gather, lane-per-key indexer, window hint, RDNA3.5 MMQ configs |
| Strata (Niko1221/Strata) | MIT | adaptive VRAM expert cache, concurrent cold-expert compute, MTP + prompt-lookup drafting |
| vLLM / sglang | Apache-2.0 | design only: scheduler, hybrid KV manager, radix cache with state checkpoints |
| radiance (libr4d) | Apache-2.0 | block-scaled fp8 WMMA fragment map (via dsv4-engine) |

## Milestones

1. **Foundation**: flake (TheRock 10.1 SDK, fat gfx1201+gfx1151), `unholy-probe`
   (topology + verified transports + signalling latency), weight loader (native
   safetensors + EXL3), memory planner dry-run.
2. **Correct forward**: single sequence across all three devices, paged caches from day one,
   validated per layer and end-to-end; quant study (EXL3 vs MXFP4).
3. **Serving core**: scheduler, cache manager, prefix cache, OpenAI server, concurrency tests.
4. **Speed**: kernels to the budget above; device-side signalling; HIP graphs; MTP/DSpark.
5. **Ship**: NixOS module and service on son-of-anton; litellm/bifrost route.
6. **Vision**.

# GLM-5.3-Flash: quant math and recommendation

Date: 2026-10-22 (revised the same day: exact DSV4 baseline, dense format, pooled indexer
layout, speculative-decoding state, quality gate).
Purpose: GLM-5.3-Flash as a **second serving mode** next to DSV4, not a replacement. It is
worth building only if the GLM build that fits this machine is better than the native DSV4
served today on the owner's workloads. This file says which builds fit, what they cost, and
what has to be measured to decide. Everything below is either measured on son-of-anton or
computed from the published checkpoint headers (method in "How the weight numbers were
produced").

## Recommendation

**Gate before porting.** The port is close to a second engine (see "Implementation notes"),
so the first measurements decide whether to do it at all:

1. Full-precision GLM-5.3-Flash (Z.ai's API or another host of the full model) against the
   native DSV4 served here, on the owner's workloads: the GSM8K harness (`/scratch/evalctl.sh`)
   plus code, tool calls and long context. GSM8K alone is near its ceiling (DSV4 flexible
   0.98). If full-precision GLM does not clearly win, stop: a 3-4 bit build will not beat its
   own full precision.
2. If it wins, the build we can run must win too. Two ways to measure one before porting, both
   untested on son-of-anton and both needing the DSV4 service stopped:
   - exllamav3 itself with the 3.05 / 4.05 branches (turboderp's quants are made for it and its
     ROCm path has gfx1151 tables, but a split over both cards and the APU is untested);
   - Project Maya serving Maya-L (156.3 GB GGUF, experts ~3.7 bits: the only Maya quant that meets
     >= 3 bpw) on HIP behind an OpenAI-compatible API. A proxy, not the EXL3 build we would ship
     (Maya runs GLM's EXL3 builds on NVIDIA only); see "Outside measurements".

   Failing both, only the engine port can answer, with KL against the FP8 oracle as the early
   proxy.

**Candidate builds** (experts x dense format; details in "Dense format" and the fit tables):

| build | experts | dense (attention, shared, head) | weights | decode read/token | 4 x 512k headroom | KL code / chat (measured) |
|---|---|---|---:|---:|---:|---:|
| **A** | EXL3 3.05 | EXL3 from the same branch (5-bit attention) | 125.15 GB | 9.00 GB (0.80x DSV4) | +47.8 GB | 2.72e-2 / 1.37e-1 |
| **B** | EXL3 4.05 | EXL3 from the same branch (6-bit attention) | 165.05 GB | 10.98 GB (0.98x) | +7.9 GB | 1.27e-2 / 7.06e-2 |
| **C** | EXL3 3.05 | native FP8, KDA projections re-quantized to fp8 | 129.36 GB | 12.58 GB (1.12x) | +43.6 GB | 2.84e-2 / 1.34e-1 |
| **D** | EXL3 4.05 | native FP8, KDA projections re-quantized to fp8 | 168.32 GB | 13.63 GB (1.22x) | +4.6 GB | 1.33e-2 / 6.57e-2 |

- **A leads on the numbers**: the lightest per token, ~32% of experts on the cards (served
  DSV4: 26%), ~48 GB of headroom at 4 x 512k, and 4 x 1M fits. Experts at 3.01 bits meet the
  >= 3 bpw requirement with little margin.
- **B / D are the quality options.** They fit 4 x 512k (the requirement) but not 4 x 1M, and
  they load the APU more: to send no more cold-expert bytes per token to the APU than DSV4 does
  today, 4.05 experts need ~62% of the routing on the cards from 22-24% of the experts (DSV4
  gets 53% from 26% with a static profile; Project Maya measured 61-84% from 21% with an adaptive
  cache). Choose 3.05 vs 4.05 experts by the head-to-head eval, not by footprint.
- **The dense format is open** (A/B vs C/D): the engine has no EXL3 dense path, and GLM's own
  FP8 release keeps the KDA projections in BF16. See "Dense format".

Excluded:

- **2.05 bpw** (85.13 GB): experts at 2.01 bits, below DESIGN's hard requirement (>= 3 bits).
- **BF16** (642.6 GB).
- **Native FP8 resident** (328.33 GB). SSD streaming it is far below the targets; see the
  streaming section.
- **The FP8 release's dense as-is** (build E: 3.05 experts + BF16 KDA): 17.14 GB per token
  (1.53x DSV4). Only worth it if fp8 KDA projections measurably hurt.

## Measured quality: KL against the FP8 oracle (G4, 2026-10-24)

unholy's GLM reference forward (`src/model/glm5*`, `src/kernels/ref_glm.*`; `tools/box/glm-ref.sh`), layer-streamed
on one R9700, against the transformers oracle on the FP8 release (`tools/ref/glm_oracle`, eager attention):
1024-token prompt + 8 teacher-forced decode positions, KL(oracle || build) and top-1 agreement over the 1032
positions. Two prompts: **code** (unholy's own sources, `mkids.py`) and **chat** (GSM8K test questions and their
worked answers as user / assistant turns through GLM's template, `mkchat.py`).

| | experts | dense | KDA projections | code KL | code top-1 | chat KL | chat top-1 |
|---|---|---|---|---:|---:|---:|---:|
| oracle eager vs SDPA (noise floor) | fp8 | fp8 | bf16 | 3.42e-3 | 97.87% | 2.25e-2 | 96.41% |
| reference path, the release as-is | fp8 | fp8 | bf16 | 3.10e-3 | 98.64% | 2.55e-2 | 96.03% |
| same, fp8 activations (W8A8) | fp8 | fp8 | bf16 | 5.50e-3 | 97.77% | 2.89e-2 | 96.61% |
| **A** | EXL3 3.05 | EXL3 (5-bit) | EXL3 5-bit | 2.72e-2 | 95.06% | 1.37e-1 | 91.18% |
| **B** | EXL3 4.05 | EXL3 (6-bit) | EXL3 6-bit | 1.27e-2 | 96.71% | 7.06e-2 | 93.22% |
| **C** | EXL3 3.05 | fp8 | fp8 (ours) | 2.84e-2 | 95.83% | 1.34e-1 | 91.38% |
| **D** | EXL3 4.05 | fp8 | fp8 (ours) | 1.33e-2 | 96.61% | 6.57e-2 | 93.31% |
| E3 | EXL3 3.05 | fp8 | bf16 | 2.70e-2 | 95.74% | | |
| E4 | EXL3 4.05 | fp8 | bf16 | 1.25e-2 | 97.00% | | |
| C, fp8 activations | EXL3 3.05 | fp8 | fp8 (ours) | 2.86e-2 | 95.64% | | |
| D, fp8 activations | EXL3 4.05 | fp8 | fp8 (ours) | 1.51e-2 | 95.83% | | |

- **The reference path is at the oracle's noise floor** (code: 3.10e-3 against eager-vs-SDPA's 3.42e-3; chat:
  2.55e-2 against 2.25e-2: the chat's answers are far less predictable than code, so the same arithmetic noise
  costs more KL there, and so does quantization). Chunked prefill (256-token chunks) and two sequences batched
  together give bit-identical logits; the pooled indexer's selection matches the oracle's on a 4096-token prompt
  (every query that sees all keys identical; past 2048 keys mean Jaccard 0.9996, at most 2 of 512 pools swapped
  at near-ties).
- **The expert width decides the quality**: 4.05 experts halve the KL of 3.05 on both prompts and gain 1.5-2
  points of top-1.
- **The dense format hardly matters**: A vs C and B vs D are within a few percent, and re-quantizing the KDA
  projections to fp8 costs ~5% over keeping them in bf16 (C vs E3, D vs E4). Choose it on speed and memory.
- **fp8 activations cost**: +2.4e-3 / +3.4e-3 (code / chat) on the release itself, +14% on D (code). A
  weight-only fp8 path (W8A16: dequantize in the GEMV / GEMM, activations in bf16) avoids it; the decode GEMV is
  bound by weight bytes either way.
- Scale, not a verdict (KL across models only gives a scale; the head-to-head evals decide): DSV4-VE as served
  is its own release precision (4.19e-3 / 98.26% against its oracle on the same code prompt, its noise floor),
  and its EXL3 3.04 experts measured 2.4-2.8e-2 / 92.7% on the earlier code prompt. A / C sit where that build
  sat, B / D at half of it.

## The reference point: what son-of-anton serves today

The installed `unholy.service` serves the **native** DSV4 checkpoint
(`/scratch/dsv4-ve`, no `--exl3`), 157 GiB on disk, with:

```
unholy serve --model /scratch/dsv4-ve --host 0.0.0.0 --port 8200 \
  --served-model-name dsv4-flash --kv-tokens 1572864 --max-model-len 786432 \
  --max-num-seqs 8 --max-batch-tokens 4096 --prefix-snapshots 64 --spec-drafts 5
```

`unholy plan` with the same geometry (`--profile data/expert_profile_unholy_mixed.txt`):

```
geometry: kv pool 1572864 tokens (6144 pages), up to 786432 tokens per request, 8 requests at once; step 4096 tokens; experts native (MXFP4)
card 0 (dev 0): layers 0-21  dense 4.48 GB  cache 4.49 GB  scratch 3.35 GB  hot experts 1471 (19.67 GB of 19.67 GB budget)
card 1 (dev 1): layers 22-42  dense 5.10 GB  cache 4.92 GB  scratch 3.47 GB  hot experts 1383 (18.49 GB of 18.50 GB budget)
apu (dev 2): cold experts 109.01 GB of 129.53 GB free; predicted routing on cards 53.0%
vision tower: 1.20 GB on the APU
dspark drafter: 0.74 GB on card 1 (in its dense), 10.27 GB of experts on the APU
```

Derived budget:

| item | GB |
|---|---:|
| cards usable (2 x 32 GB; 34.21 GB physical each) | 64.00 |
| APU free for the model | 129.53 |
| **total usable** | **193.53** |
| served weights | 168.22 |
| served KV cache (1,572,864 tokens) | 9.41 |
| scratch (both cards) | 6.82 |
| **served total** | **184.45** |
| **headroom** | **9.08** |

DSV4 KV is ~5.98 KB/token (compressed CSA/HCA + fp4 indexer keys).
The `--exl3` plan for comparison: cold experts 66.13 GB instead of 109.01 GB,
hot experts 2076 / 1952, predicted card routing 64.2%.

Weights read per decode token by the served DSV4, from its headers (43 layers; the embedding
lookup, MTP/DSpark and vision excluded):

| group | GB |
|---|---:|
| attention (fp8: wq_a, wq_b, wkv, wo_a, wo_b) | 4.60 |
| compressors (bf16) | 0.53 |
| indexer | 0.28 |
| shared experts (fp8) | 1.08 |
| head (bf16) | 1.06 |
| router, HC, norms | 0.23 |
| **non-routed** | **7.77** |
| routed experts, 6 of 256 active (MXFP4) | 3.45 |
| **total** | **11.22** |

At the predicted 53% card routing, 0.47 x 3.45 = **1.62 GB of cold experts per token** go to
the APU.

No EXL3 is served. The DSV4 `--exl3` build (3.04 bpw) took only the routed experts from EXL3;
attention, shared experts and the rest stayed native fp8 (DECISIONS 2026-10-11, `exl3_experts`
in `src/model/dsv4.hpp`). The engine has no EXL3 path for dense weights.

## GLM-5.3-Flash shape (from the released config and the FP8 headers)

- 321.323B parameters; **16.7B active per token** in the main model (embedding and MTP
  excluded). `hidden_size` 4096, 45 layers + 1 MTP layer.
- 3 dense MLP layers (0-2, intermediate 12288), 42 sparse MoE layers: 288 routed experts,
  top-8, 1 shared expert, `moe_intermediate_size` 2048, `routed_scaling_factor` 2.5, sigmoid
  routing with `e_score_correction_bias` (`noaux_tc`).
- 34 `linear_attention` layers (KDA / Kimi delta rule): q/k/v 4096 -> 8192 (64 heads x 128),
  o 8192 -> 4096, short causal conv (kernel 4) on q/k/v, forget and output gates through
  rank-128 projections (`f_a`/`f_b`, `g_a`/`g_b`), per-head beta (`b_proj`), `A_log`/`dt_bias`,
  gated output norm; constant recurrent state, no per-token KV.
- 11 `deepseek_sparse_attention` layers (3, 7, ..., 43): MLA with 64 heads, `q_lora_rank`
  1536, `kv_lora_rank` 512, `qk_nope_head_dim` 256, `v_head_dim` 256, no rope
  (`mla_use_nope`); top-2048 selection through a pooled indexer (32 heads x 128,
  `index_kpool` 4, the incomplete tail pool always selected).
- mHC: `hc_mult` 4, 20 Sinkhorn iterations (the DSV4 engine already has HC).
- 1 nextn / MTP layer (45) with its own 288 experts and DSA indexer
  (`index_share_for_mtp_iteration` true). No 3-stage DSpark.
- Vision: 24 blocks, 1024 hidden, patch 14, 448 px, temporal patch 2,
  spatial merge 2, out 4096.
- Vocab 154,880; context 1,048,576; GLM tokenizer and chat template
  (`reasoning_effort` low/high/max, `clear_thinking`), not the DSV4 text layer.

Parameter split (exact, from the FP8 checkpoint headers):

| group | params | dtype in the FP8 release |
|---|---:|---|
| routed experts (42 layers + MTP) | 311.65B | fp8 e4m3, 128x128 block scales |
| KDA (34 layers) | 4.68B (q/k/v/o 4.56B) | **BF16** |
| DSA q_a / q_b / kv_a / o (11 layers) | 1.11B | fp8 |
| DSA kv_b + indexer | 0.27B | BF16 |
| shared experts | 1.06B | fp8 |
| dense MLP (layers 0-2) | 0.45B | fp8 |
| embed / head | 0.63B / 0.63B | BF16 |
| vision | 0.56B | BF16 |
| router, HC, norms, MTP non-expert | 0.27B | BF16 / fp32 |

## How the weight numbers were produced

The public branches were read byte-exactly from their safetensors headers, not
from the model card:

1. List the branch's shard files with the HF tree API.
2. For each file, read the 8-byte little-endian header length with an HTTP range
   request, then range-read exactly that many header bytes.
3. Sum `data_offsets[1] - data_offsets[0]` per tensor and classify by tensor name:
   `.mlp.experts.` -> routed experts; `.mlp.shared_experts.` -> shared experts;
   `.mlp.gate.` -> router; other `.mlp.` -> dense MLP; `.self_attn.indexer.` -> indexer;
   `.self_attn.kv_b_proj` -> kv_b; other `.self_attn.` -> KDA or DSA by the config's
   `layer_types`; `embed_tokens` -> embed; `lm_head` -> head; `hc_` -> hyper-connections;
   `norm` -> norms; `model.visual.*` -> vision; `layers.45.*` -> MTP layer.
4. The sums agree with the tree file sizes to within ~0.1 GB (headers, index,
   configs, and `kpool_aux.safetensors` are excluded).

Read this way: EXL3 branches `4.05bpw`, `3.05bpw`, `2.05bpw` (each includes the vision tower
and the MTP layer; the MTP experts are inside the routed-expert number), all 62 shards of the
native FP8 checkpoint (dtypes and shapes exact), and the native DSV4 checkpoint's headers
locally on son-of-anton. In the EXL3 branches the KDA q/k/v are fused into `qkv_proj`; the
trellis shape gives each linear's bits.

## Weight bytes and effective precision

Weights only, decimal GB ("other" = dense MLP, router, HC, norms, MTP non-expert):

| | routed experts | KDA | DSA main | kv_b + indexer | shared | head | embed | vision | other | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **EXL3 4.05bpw** | 156.28 | 3.66 | 0.83 | 0.45 | 0.79 | 0.48 | 1.27 | 0.60 | 0.68 | **165.05** |
| **EXL3 3.05bpw** | 117.33 | 3.09 | 0.69 | 0.43 | 0.66 | 0.48 | 1.27 | 0.60 | 0.60 | **125.15** |
| native FP8 | 311.73 | 9.37 | 1.11 | 0.53 | 1.06 | 1.27 | 1.27 | 1.13 | 0.87 | **328.33** |
| MXFP4 re-quant + 4.05 non-experts | 165.56 | 3.66 | 0.83 | 0.45 | 0.79 | 0.48 | 1.27 | 0.60 | 0.68 | **174.33** |

EXL3 2.05bpw: 85.13 GB, of which 78.37 GB routed experts (2.01 bits). BF16 is
321.3B x 2 bytes = 642.6 GB.

Bits per linear in the EXL3 branches (from the trellis shapes):

| branch | routed experts | KDA qkv / o, DSA q_a / q_b / kv_a / o, indexer wq_b | shared | dense MLP | head |
|---|---:|---:|---:|---:|---:|
| 4.05 | 4 | 6 | 6 | 5 | 6 |
| 3.05 | 3 | 5 | 5 | 4 | 6 |

Both branches keep `kv_b_proj`, the KDA gate projections (`f_*`, `g_*`, `b_proj`), the
indexer's `wk` / `weights_proj` and the router in F16, and the embedding in BF16. Effective
bits by group: experts 4.01 / 3.01, KDA 6.26 / 5.29 (F16 gates included), DSA main 6.01 /
5.01, all non-experts 7.25 / 6.47.

Sizes that matter for download/space:

- 4.05 branch: 165.15 GB on the HF tree (19 shards + `mtp.safetensors`).
- 3.05 branch: 125.26 GB (15 shards + `mtp.safetensors` + `kpool_aux.safetensors`).
- 2.05 branch: 85.23 GB (12 shards).
- native FP8: 62 shards, 328.33 GB of tensors (index `total_size` 328,326,771,576).
- The `sc_*` variants are 4.10/3.10/2.60/2.10 -> 167.81 / 128.03 / 109.00 / 88.28 GB.
  They use exllamav3 1.6.x and `out_scales: auto`; the "sc" meaning is undocumented,
  so they are not used here.

## Dense format

The engine runs DSV4's dense weights in their native fp8 (128x128 block scales) through the
W8A8 kernels (decode GEMV 540-556 GB/s on 32 MB matrices, DECISIONS 2026-10-14); EXL3 exists
only for routed experts. DSV4 was trained in that fp8, so the path is lossless. GLM has no
lossless option that fits:

| option | KDA projections | dense on the cards (incl. embed, MTP) | non-routed read/token | kernels | vs BF16 |
|---|---|---:|---:|---|---|
| EXL3, 3.05 branch (A) | 5-bit EXL3 | 7.22 GB | 5.82 GB | new: EXL3 dense GEMV/GEMM (the expert kernels are the start) | unmeasured |
| EXL3, 4.05 branch (B) | 6-bit EXL3 | 8.17 GB | 6.74 GB | same | unmeasured |
| FP8 release, KDA re-quantized (C/D) | fp8, our quantization | 10.91 GB | 9.39 GB | existing W8A8 path | unmeasured; Z.ai left KDA in BF16 |
| FP8 release as-is (E) | BF16 | 15.47 GB | 13.96 GB | bf16 GEMV | lossless |

- The DSA projections and shared experts are fp8 in the release, so C/D are lossless there;
  only the KDA projections (4.56B parameters) are ours to quantize.
- Bytes favour EXL3, kernels favour fp8. At decode they break even if the EXL3 dense GEMV
  moves ~340 GB/s of its smaller weights against fp8's ~550 GB/s (5.82 vs 9.39 GB per token).
  The EXL3 GEMV's throughput on dense shapes has not been measured.
- Weight-space error of 5/6-bit EXL3 and of fp8 against the BF16 KDA projections can be
  measured before choosing, as DECISIONS 2026-10-11 did for DSV4 (there, 5-bit EXL3 attention
  was rel L2 0.038-0.041 against the fp8 weights).
- Other mixes are possible: 3.05 experts with 4.05 dense needs a merged EXL3 checkpoint or a
  loader that takes two EXL3 sources; fp8 dense with EXL3 experts is what the existing
  `--model` (native) + `--exl3` (experts) split already expresses.

## KV and per-sequence state

**Pooled indexer rows.** `Glm5NextTextIndexer.get_pooled_states` builds each pool's key as a
weighted sum of its 4 tokens' keys, the weights a softmax over the 4 tokens of (gate + ape) taken
separately for each of the 128 key dimensions (the gate is a 128-wide projection of the hidden
state; Project Maya's parity harness pins the same quirk), and a pool is selectable only once it
is complete. A pool's key therefore depends only on its own 4 tokens and can be stored once,
when the pool completes (DSV4's compressor pattern). The incomplete tail (<= 3 tokens' keys and
gates per layer) is a per-sequence carry, and its tokens are always selected raw. The HF cache
instead keeps every token's key, gate and valid flag and re-pools on every call.

Per token (scales excluded, < 1%):

| | pooled rows (used here) | HF per-token layout |
|---|---:|---:|
| 11 DSA layers, latent KV, fp8 (11 x 512) | 5,632 B | 5,632 B |
| 11 indexers | 352 B (11 x 128 / 4, fp8) | 1,414 B (key + gate + valid, fp4) |
| **total** | **5,984 B** | 7,046 B |
| with bf16 latent | 11,616 B | 12,678 B |

The pooled layout costs the same per token as DSV4 (~5.98 KB). It also matters at decode: at
512k depth the indexer reads 0.19 GB per sequence per step (fp8 pooled) instead of 0.74 GB
(per-token, fp4); DSV4 reads 0.18 GB. fp4 pooled rows would halve the index (176 B/token).

Per sequence, KDA state:

- recurrent state: 34 layers x 64 heads x 128 x 128 x 4 B (fp32) = **142.6 MB/seq**
- conv state: 34 x 24,576 channels x 3 taps x 4 B ~= **10.0 MB/seq**
- **~152.6 MB/seq**, ~22x DSV4's per-slot state (~7 MB).

**Speculative decoding.** A verify step that rejects drafts must leave the state at the last
accepted token. Planned here: double-buffer it (keep the pre-step state, write the new one
beside it; after acceptance either keep the new state or replay the accepted tokens from the
old one), **305.3 MB per sequence**. Project Maya saves the recurrent states before a verify and
restores them on a miss the same way, token-identical to its plain decode loop. A state per draft
position instead would cost
6 x 152.6 = 916 MB per sequence at `--spec-drafts 5` (7.3 GB at 8 sequences). Every step also
reads and writes the state: 0.31 GB of traffic per sequence, which the batch does not share.

Totals (KV + double-buffered state):

| pool | KV, fp8 latent | state x 2 | **total** | with bf16 latent |
|---|---:|---:|---:|---:|
| 1,572,864 tokens (served geometry), 8 seq | 9.41 | 2.44 | **11.85** | 20.71 |
| 2,097,152 tokens (4 x 512k), 4 seq | 12.55 | 1.22 | **13.77** | 25.58 |
| 4,194,304 tokens (4 x 1M), 4 seq | 25.10 | 1.22 | **26.32** | 49.94 |

DSV4 was trained with fp8 KV; whether GLM's latent tolerates fp8 is unmeasured (HF keeps it
in bf16).

**Important for the cache manager:** the KDA state is unpaged per-sequence state. Prefix-cache
snapshots can no longer be ~7 MB each: 64 snapshots at 152.6 MB would be 9.8 GB of pinned
host memory. Either keep far fewer GLM snapshots, store them at fewer boundaries,
or compress the state (bf16 halves it but changes bits). This is a design decision
for the port, not a quant decision.

## Fit

Capacity 193.53 GB; scratch assumed to stay 6.82 GB (GLM's could be larger: top-8 MoE carries
33% more expert rows per token than DSV4's top-6, plus KDA chunk buffers). KV + state from the
table above (fp8 latent, pooled fp8 index, double-buffered KDA state). These are
planner-equivalent projections: the engine cannot load GLM yet.

| build | weights | served pool, 8 seq | 4 x 512k | 4 x 1M |
|---|---:|---:|---:|---:|
| served DSV4 native | 168.22 | 184.45 (**+9.08**) | | |
| **A** EXL3 3.05 | 125.15 | 143.82 (**+49.71**) | 145.74 (**+47.79**) | 158.29 (**+35.24**) |
| **B** EXL3 4.05 | 165.05 | 183.72 (**+9.81**) | 185.64 (**+7.89**) | 198.19 (**-4.66**) |
| **C** fp8 dense + 3.05 | 129.36 | 148.03 (**+45.50**) | 149.95 (**+43.58**) | 162.50 (**+31.03**) |
| **D** fp8 dense + 4.05 | 168.32 | 186.99 (**+6.54**) | 188.91 (**+4.62**) | 201.46 (**-7.93**) |
| E bf16 KDA + 3.05 | 133.92 | 152.60 (+40.93) | 154.51 (+39.02) | 167.06 (+26.47) |
| MXFP4 + 4.05 non-experts | 174.33 | 193.00 (+0.53, on the edge) | | |

Conclusions:

- **4 x 512k is the requirement, and every candidate meets it.** 4 x 1M is a bonus that only
  the 3.05-expert builds have.
- B and D have 5-10 GB of headroom, like the served DSV4; GLM's unmeasured scratch comes out of
  that first.
- bf16 latent KV (+8.86 GB at the served pool) leaves B at +0.95 GB and does not fit D; the
  3.05-expert builds take it easily.

## Hot experts and APU load

Card expert budget at the served geometry = 64 GB - dense - KV/state (11.85) - scratch (6.82):

| build | expert size | card budget | experts on cards | share | APU cold | share at 4 x 1M |
|---|---:|---:|---:|---:|---:|---:|
| A | 9.47 MB | 38.10 GB | ~4,020 | 32.5% | 79.2 GB | 20.1% |
| B | 12.62 MB | 37.16 GB | ~2,945 | 23.8% | 119.1 GB | (does not fit) |
| C | 9.47 MB | 34.42 GB | ~3,630 | 29.3% | 82.9 GB | 17.0% |
| D | 12.62 MB | 34.42 GB | ~2,725 | 22.0% | 121.9 GB | (does not fit) |
| served DSV4 native | 13.37 MB | 38.16 GB | 2,854 | 25.9% of 11,008 | 109.0 GB | |

Shares are of 12,384 experts (43 MoE layers incl. MTP x 288). A share of experts is not a share
of routing. STATUS: multi-stream decode and prefill are bound by the cold experts on the APU
(65-82% busy), and the APU's load per token is (1 - r) x the active routed bytes (and FLOPs),
r being the routing share that lands on the cards. Served DSV4: 1.62 GB per token. GLM's
active routed bytes are 3.18 GB (3.05) / 4.24 GB (4.05), 8 experts instead of 6, so matching
DSV4's APU load needs **r >= 49% with 3.05 experts and r >= 62% with 4.05 experts**. DSV4
reaches 53% with 26% of its experts on the cards (a static mixed-traffic profile; dsv4-engine's
code-only calibration caught 85% with 32%). For GLM, Project Maya measured an adaptive cache at
74-82% of the routing with 31% of the experts and 61-84% with 21%, on short single prompts (see
"Outside measurements"): well above A's and C's 49%, around B's and D's 62%. A static profile over
mixed traffic will catch less, so our own profile is still needed (the oracle run can produce
one). Routed FLOPs per token: 16.9 G vs DSV4's 13.0 G (+30%).

## Decode bytes per token

Weights read per decode token (main model; embedding lookup, MTP and vision excluded):

| build | non-routed | routed, 8 of 288 active | total | vs DSV4 |
|---|---:|---:|---:|---:|
| served DSV4 native | 7.77 | 3.45 (6 of 256, MXFP4) | **11.22** | 1.00 |
| A | 5.82 | 3.18 | **9.00** | 0.80 |
| B | 6.74 | 4.24 | **10.98** | 0.98 |
| C | 9.39 | 3.18 | **12.58** | 1.12 |
| D | 9.39 | 4.24 | **13.63** | 1.22 |
| E | 13.96 | 3.18 | **17.14** | 1.53 |

Per sequence and step on top of the weights: the KDA state, 0.31 GB (read + write) at any
depth, and the indexer, 0.19 GB at 512k (pooled fp8). DSV4: 0.18 GB at 512k plus a few MB of
rings.

Expectation: the served DSV4 decodes one stream at 34.2 ms/token at 4k (29 tok/s) and
44.8 ms at 512k (22 tok/s) without speculation (STATUS). On bytes, A and B should land near
that if the EXL3 dense GEMV keeps pace (see "Dense format"), C/D ~10-20% slower. Bytes are
first order only; the multi-stream bound is the APU (above). The single MTP layer is not the
3-stage DSpark, so do not assume DSV4's draft acceptance (3.1 tokens per verify step on code).
Project Maya drafts one token per step with it, pipelined (the next draft runs while the current
token finishes): 83-86% of drafts accepted in live serving, 97% in greedy runs, decode 26.2 ->
40.0 tok/s on 2x V100.

## Outside measurements: Project Maya

[Project Maya](https://github.com/mw00/project-maya) (MIT; derived from Strata, which unholy
already credits) runs GLM-5.3-Flash with its experts tiered over VRAM, RAM and SSD, mostly as GGUF
i-quants. Its numbers are self- or community-reported, on other machines, from a repo created
2026-10-06: they set scales, not decisions.

**Routing** (`bench/results/2026-10-05-glm-single-gpu`: one V100, an adaptive LFU cache warmed
from a prior, three 128-token requests on chat / code / math prompts, 4k context):

| experts in VRAM (of 12,096) | share of experts | VRAM hits |
|---:|---:|---:|
| 714 | 5.9% | 29-50% |
| 1,302 | 10.8% | 44-65% |
| 2,562 | 21.2% | 61-84% |
| 3,738 | 30.9% | 74-82% |
| 8,458 (2 x V100) | 69.9% | 98% |

Replaying 1000 tokens of routes at the 2 x V100 size, LRU and LFU reach 97.7 / 96.4% per GPU and
Belady's optimum 98.6 / 98.0%. Their `data/expert-profile-glm.bin` ranks all 46 x 288 experts (MTP
layer included) but carries no counts: it can seed a placement, not give r.

**Speculative decoding** (`bench/results/2026-10-06-glm-spec`, 2 x V100, after a 500-token prompt):
one MTP draft per step, pipelined, recurrent states restored on a miss. Plain 38.1 ms/token
(26.2 tok/s); speculative 25.0 ms (40.0 tok/s) with 97% of drafts accepted in greedy runs; 83-86%
accepted in live serving; the draft equals the next greedy token 64-68% of the time on
teacher-forced doc/code text.

**Quality against the FP8 model** (`bench/results/MAYA-*.md`): KL(FP8 || quant) over the top 64
log-probs plus a remainder bucket, on 8 held-out texts (7,672 positions) through their engine;
zero-shot on 400 questions per task (ARC-E/C, HellaSwag, WinoGrande, PIQA), lm-evaluation-harness
scoring, the FP8 model run layer by layer in PyTorch.

| quant | size | ~bits/weight | routed experts | KL vs FP8 | same top token | zero-shot kept |
|---|---:|---:|---|---:|---:|---:|
| Maya-L | 156.3 GB | 3.9 | IQ3_S gate/up, IQ4_XS down (Q5_K in 8 layers) | 0.188 | 90.0% | 99.2% |
| Maya-M | 116 GB | 2.9 | IQ2_S gate/up, IQ3_XXS down (IQ3_S in 8 layers) | 0.29-0.33 | 86-87% | 97.9% |
| Maya-S | 96.5 GB | 2.4 | IQ2_XXS gate/up, IQ2_S down (IQ3_XXS in 8 layers) | 0.428 | 83.3% | 97.9% |

All three keep attention, shared experts and dense layers at Q6_K (KDA gate projections and the
indexer Q8_0, the router F32). Maya-S's KL by text runs from 0.06 (a reasoning trace) to 0.85 on
tool calls and 0.93 on wikitext. Their quant plan quotes ~0.07 KL for a 3.5 -> 3.0-bit step, and Maya-L missed its own
target of KL <= 0.05. These KLs are an order of magnitude above DSV4 EXL3 3.04's 2.7e-2, but on
other texts against another reference, so they do not compare across models; within GLM the loss
grows steadily as bits drop (0.19 -> ~0.31 -> 0.43 from 3.9 to 2.4 bits), which is why the
3.05-vs-4.05 choice needs its own eval. Maya does not measure turboderp's 3.05 / 4.05 branches.

**Speed on these devices** (README and `docs/AMD_MAYA.md`; community and maintainer reports):

| machine | quant | decode | prefill |
|---|---|---:|---:|
| 2x R9700 (PCIe 4.0 x8 each), Ryzen 5 3600, 64 GB RAM, 256K context | Maya-S24 (94.7 GB) | 44.1 tok/s | 704 tok/s |
| same | Maya-M-Derisked (116 GB, part of it in RAM) | 24.9 tok/s | 617 tok/s |
| 1x R9700, 128 GB RAM | Maya-S | 21.4 tok/s | |
| Strix Halo, 128 GB, every expert in the GPU pool | Maya-S | ~17.8 tok/s | ~210 tok/s (3-4K prompts) |
| Gorgon Halo (Radeon 8065S), 192 GB, MTP decode | Maya-S | 28.4-29.6 tok/s | 271-278 tok/s |

**Code worth reading for the port:** `src/kernels/cuda/glm_kda.cu` (conv, lower-bounded gate,
recurrence, gated norm), `glm_dsa.cu` (pool, score, select with the tail, absorbed MLA),
`glm_hc.cu`; the float64 reference `ref/glm.py`; the FP8 zero-shot scripts
`tools/maya_quant/zs_*.py`; a synthetic GLM GGUF with per-op dumps (`data/glm5-synth.gguf`,
`data/glm5-synth-dumps12/`) as test vectors. The kernels are f32 and reference-grade (one block
per head stepping through the tokens, O(n^2) selection), built for gfx1201 / gfx1151 and passing
their parity tests on an R9700. Their EXL3 on HIP is FP32-FMA bring-up, not a donor.

## Prefill at 512k depth

FLOPs per prefill token at depth d = 512k:

- weights: 2 x 16.7B active = 33.5 G (DSV4: 27 G)
- indexer: 11 layers x d/4 pools x 32 heads x 128 x 2 = 22,528 d -> **11.8 G** (DSV4: 45 G)
- sparse attention: 11 x 2,051 rows x 64 heads x 1,024 x 2 -> 3.0 G
- KDA, chunked (chunk 64): ~0.25 G, independent of depth
- **~48.5 G/token: 500 tok/s = ~24 TFLOPS sustained** (DSV4: ~85 G and ~42 TFLOPS; it measures
  ~900 tok/s marginal at 491k)

On paper the budget is easier than DSV4's. The risks: the KDA chunked-prefill kernels are new;
the experts are a larger share of the work (+30% routed FLOPs, more of it on the APU with 4.05
experts); and attention gathers 2,051 raw rows per query.

## SSD streaming: native FP8 and MXFP4

Measured NVMe on son-of-anton (WD Blue SN5100 4TB, `/scratch`):

| method | throughput |
|---|---:|
| O_DIRECT, 8M blocks | 5.8 GB/s |
| page-cache read, 8M blocks | 6.3 GB/s |
| O_DIRECT, 1M blocks | 3.5 GB/s |

The model is read once per prefill chunk (every expert is used by a 4096-token
chunk), and once per active expert per decode token. That gives:

```
prefill tokens/s <= chunk_tokens * SSD_BW / cold_expert_bytes
decode tokens/s  <= SSD_BW / (active_routed_bytes * cold_routing_fraction)
```

**Native FP8 (328.33 GB):**

- Resident expert capacity with EXL3 4.05 non-experts:
  hot ~37.2 GB on cards + 128.9 GB on the APU = ~166.1 GB of 311.65 GB.
  Cold ~145.6 GB.
- Prefill ceiling at 4096-token chunks: `4096 * 5.8 / 145.6 = 163 tok/s`.
  At 1024-token chunks: `1024 * 5.8 / 145.6 = 41 tok/s`.
- Decode: active routed FP8 = 8.46 GB/token. Cold routing fraction ~0.3-0.4
  (the hot set captures most of the mass): 2.5-3.4 GB/token -> 438-583 ms/token
  -> **1.7-2.3 tok/s**. Uniform 47% gives 1.5 tok/s.
- With the FP8 release's own dense (BF16 KDA) the cold set is ~153 GB and the prefill
  ceiling ~155 tok/s; conclusions unchanged.

This cannot meet the project targets (500 tok/s prefill, 20 tok/s decode). It is a
research baseline, not a serving mode.

**MXFP4 re-quant + EXL3 4.05 non-experts (174.33 GB):**

- Resident capacity ~166 GB of 165.56 GB experts: essentially all experts fit.
  At the full 1,572,864-token pool the total is 193.00 GB vs 193.53 GB capacity
  (+0.53 GB, too tight to rely on once GLM's scratch is known). At 1M tokens it is
  189.87 GB (+3.7 GB).
- If a few GB are cold, the SSD cost is small: active routed MXFP4 is 4.49 GB/token,
  and a 2-5% cold fraction is 0.09-0.22 GB/token -> 15-38 ms/token of SSD time.
  Prefill cold bytes per 4096 chunk are ~GB, so prefill is not SSD-bound.
- Requires building an MXFP4 quantizer for GLM's FP8 experts; GLM's experts were not
  trained in MXFP4 (DSV4's were), so its quality vs the calibrated EXL3 4.05 is unmeasured.
  Not needed for the main path.

Space is no longer an issue: `/scratch` has 1.8T free (was 230G before cleanup).
Native FP8 (332 GB of shards) plus 3.05 (125 GB) and 4.05 (165 GB) all fit.

## Implementation notes for the next agent

The port is the large part, and it starts only if the gate passes. GLM is not a DSV4 variant:

1. **KDA layers (34).** Chunked/recurrent Kimi delta rule with q/k/v projections
   (64 heads x 128), short causal conv (kernel 4), forget gate, per-head beta,
   gated output norm, fp32 recurrent state of 64x128x128 per layer per sequence,
   double-buffered for speculative decoding. New kernels and a new state layout; no existing
   DSV4 kernel covers this.
2. **DSA layers (11).** MLA without rope (64 heads, q_lora 1536, kv_lora 512, nope 256,
   v 256), top-2048 selection = 512 pools of 4 plus the raw tail (<= 3 tokens), pooled indexer
   (32 heads x 128, `index_kpool_compress_ape/gate`). `indexer_types` is all "full" here, so no
   cross-layer top-k sharing is needed initially, but keep the hook.
3. **mHC (4 streams) and MoE.** 288 experts, top-8, shared expert, sigmoid routing
   with correction bias and routed scale 2.5; the existing DSV4 MoE/EXL3/MXFP4
   kernels are the starting point, but the expert count and router differ.
4. **MTP (1 layer).** Layer 45 has 288 experts and its own DSA indexer; it is not
   the 3-stage DSpark drafter. Speculative decoding support is a separate design.
5. **Vision.** 24 blocks, 1024 hidden, patch 14, temporal patch 2, spatial merge 2,
   448 px; the EXL3 branches carry it at 0.60 GB. The DSV4 ViT path is a template,
   not a drop-in.
6. **Text layer.** GLM tokenizer (154,880) and chat template with
   `reasoning_effort` low/high/max and `clear_thinking`. The DSV4 tokenizer/chat
   code is not reusable as-is.
7. **Cache manager.** DSA latent pages (1 row per token x 512, fp8 + block scales);
   pooled indexer rows written when a pool completes, plus a per-sequence tail carry;
   KDA state double-buffered (305 MB/seq). Prefix caching needs a new snapshot
   budget/strategy.
8. **Dense format.** EXL3 dense kernels (A/B) or KDA fp8 re-quantization at load (C/D);
   decide with the measurements in "Dense format".
9. **Planner.** 43 MoE layers (42 + MTP) x 288 experts; expert bytes per format
   12.62 / 9.47 / 25.17 / 13.37 MB for EXL3 4.05 / 3.05 / FP8 / MXFP4;
   vision on the APU; no DSpark experts. The planning code is otherwise format-agnostic.
10. **Serve geometry.** For parity with the served DSV4 use
    `--kv-tokens 1572864 --max-model-len 786432 --max-num-seqs 8`.
    The DESIGN target (4 x 512k) is `--kv-tokens 2097152 --max-model-len 524288 --max-num-seqs 4`;
    4 x 1M (`--kv-tokens 4194304 --max-model-len 1048576`) fits only with 3.05 experts.
11. **Verification.** There is no DSV4-style oracle for GLM in the tree. Use the
    HF transformers `glm5_next` implementation running the native FP8 checkpoint
    as the oracle, layer-streamed on one card, the same way `tools/ref/oracle`
    does for DSV4. Then compare per-layer and end-to-end KL/top-1. Have it record the
    router's choices too: that histogram is the first GLM routing profile. Project Maya's
    `ref/glm.py` and synthetic dumps are a second, independent check of the layer math.

## Open questions / next measurements

In order:

1. **Gate:** full-precision GLM vs the served DSV4 on the owner's workloads (see
   "Recommendation"). Nothing else matters if this fails.
2. **The runnable build:** exllamav3 on son-of-anton with the 3.05 / 4.05 branches (needs the
   DSV4 service stopped); if it runs, the same eval.
3. **Quality vs FP8:** measured 2026-10-24 on a code and a chat prompt ("Measured quality"): 4.05 experts
   halve 3.05's KL, the dense format hardly matters, fp8 activations cost. Still open: long context
   (the 4096-token run only checked the indexer's selection) and prose beyond the GSM8K chat.
4. **Routing profile:** r for each build from the oracle's router histogram (break-even with
   DSV4's APU load: 49% for 3.05 experts, 62% for 4.05). Maya's adaptive-cache hit rates suggest
   the 3.05-expert builds clear it and the 4.05-expert builds are marginal.
5. **Dense format:** weight-space error of 5/6-bit EXL3 and of fp8 against the BF16 KDA
   projections; EXL3 GEMV throughput on dense shapes (break-even ~340 GB/s).
6. **fp8 latent KV:** whether GLM tolerates it (bf16 costs +8.9 GB at the served pool).
7. **Scratch:** GLM's step scratch for KDA + 288-expert MoE + pooled indexer is not
   measured. If it exceeds 6.82 GB, B's and D's headroom shrinks first and MXFP4 stops fitting.
8. **Prefix snapshots:** 152.6 MB per KDA state snapshot forces a different policy
   (fewer boundaries, host-resident, or compressed). Measure the cost of restoring
   a bf16 state versus recomputing.
9. **MTP yield:** 1 nextn layer. Project Maya measures 83-86% acceptance in live serving with one
   pipelined draft per step; whether chaining it for several drafts pays, and what it yields with
   this engine's batched verify, is unmeasured. Measure before committing serve defaults.
10. **`sc_*` branches:** possibly better quantizer (exllamav3 1.6.x, auto out-scales),
    undocumented calibration. Revisit only if 3.05/4.05 quality disappoints.

## Reproduction

Plan on son-of-anton:

```sh
cd /scratch/unholy
./build/unholy plan --model /scratch/dsv4-ve --profile data/expert_profile_unholy_mixed.txt \
  --kv-tokens 1572864 --max-model-len 786432 --max-num-seqs 8 --max-batch-tokens 4096 \
  --prefix-snapshots 64 --spec-drafts 5
```

The `--exl3` variant adds `--exl3 /scratch/models/dsv4-ve-exl3-3.04bpw`.

Safetensors header scan (per branch, and for the 62 FP8 shards): read `curl -L -r 0-7` for
the header length, then `curl -L -r 8-(8+len-1)` for the JSON header, and sum
`data_offsets[1]-data_offsets[0]` grouped by tensor-name class. The exact
classification rules are in "How the weight numbers were produced".

DSV4 per-token reads: the same sum over the headers of `/scratch/dsv4-ve/*.safetensors`
(read locally), grouped by `attn.` / `attn.compressor.` / `attn.indexer.` / `ffn.experts.` /
`ffn.shared_experts.` / `ffn.gate.` / `head` / `hc_`; routed active = routed x 6/256.

NVMe check (non-destructive read):

```sh
f=/scratch/dsv4-ve/model-00001-of-00048.safetensors
dd if="$f" of=/dev/null bs=8M count=256 iflag=direct
```

## Sources

- Config: https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json
- Quant branches: https://huggingface.co/turboderp/GLM-5.3-Flash-exl3
  (`4.05bpw`, `3.05bpw`, `2.05bpw`; `sc_*` variants noted above)
- Native FP8/BF16: https://huggingface.co/zai-org/GLM-5.3-Flash,
  https://huggingface.co/zai-org/GLM-5.3-Flash-BF16
- HF implementation used for the KV/state layout:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py
  (`Glm5NextTextAttention` for DSA, `Glm5NextTextLinearAttention` for KDA,
  `Glm5NextTextIndexer.get_pooled_states` for the pooled indexer)
- Model blog: https://z.ai/blog/glm-5.3-flash
- Project Maya: https://github.com/mw00/project-maya (`bench/results/2026-10-05-glm-single-gpu`,
  `bench/results/2026-10-06-glm-spec`, `bench/results/MAYA-{S,M,L}.md`, `docs/QUANT-PLAN.md`,
  `docs/AMD_MAYA.md`, README speed table), read 2026-10-23
- Served DSV4 plan: `unholy plan` on son-of-anton, 2026-10-22.

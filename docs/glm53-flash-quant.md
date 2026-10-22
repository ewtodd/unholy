# GLM-5.3-Flash: quant math and recommendation

Date: 2026-10-22.
Purpose: decide which GLM-5.3-Flash checkpoint to target before any code changes.
Everything below is either measured on son-of-anton or computed from the published
checkpoint headers (method in "How the weight numbers were produced").

## Recommendation

**Primary target: EXL3 3.05 bpw** (`turboderp/GLM-5.3-Flash-exl3`, branch `3.05bpw`).

- Experts at ~3.01 effective bits, attention at ~5.5 bits, head 6 bits.
  Same recipe as the DSV4 EXL3 3.04 build already validated in this repo.
- 125.16 GB of weights, about the same footprint as the DSV4 EXL3 plan (125.3 GB),
  and leaves ~49 GB of the machine's 193.5 GB for GLM's larger KV, a 4 x 1M pool,
  and more hot experts.
- ~31% of experts fit on the cards (vs 23.5% at 4.05), which attacks the APU
  multi-stream bound.
- Decode reads ~9.1 GB/token of weights vs ~6.9 GB/token for the served native
  DSV4, so expect roughly 20-30 tok/s at long context before tuning, with the
  single MTP layer adding drafts.

**Optional quality mix: 3.05 bpw experts + 4.05 bpw non-experts** (attention,
shared experts, dense MLP, HC, embed, head, vision). 126.09 GB total (+0.93 GB
over plain 3.05), with attention at ~6.5 bits instead of ~5.5. This is the exact
analogue of the DSV4 recipe (native/high-bit dense + EXL3 experts). It needs
either a merged checkpoint or a loader that takes dense and expert sources
separately; the current `--model` / `--exl3` split does not directly express
"EXL3 dense + EXL3 experts from a different branch".

**Fallback only: EXL3 4.05 bpw** (165.05 GB). It is the footprint match for the
native DSV4 currently served (168.22 GB) and fits the served 1.5M pool with
fp8 KV (184.2 GB total, ~9.4 GB headroom), but it cannot do 4 x 1M and has a
smaller hot set.

**Do not start at 2.05 bpw.** 85.13 GB, huge headroom, but experts at ~2.01 bits
with no evidence yet that GLM survives that.

**Do not target BF16.** 642.6 GB.

**Native FP8 (328.25 GB) is not a resident option.** SSD streaming it costs too
much throughput to meet the project targets; see the streaming section.

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

## GLM-5.3-Flash shape (from the released config)

- 321.323B parameters, ~18B active. `hidden_size` 4096, 45 layers + 1 MTP layer.
- 3 dense MLP layers (`mlp_layer_types` dense for layers 0-2), 42 sparse MoE layers.
  288 routed experts, top-8, 1 shared expert, `moe_intermediate_size` 2048,
  `routed_scaling_factor` 2.5, sigmoid routing with `e_score_correction_bias`.
- 34 `linear_attention` layers (KDA / Kimi delta rule): 64 heads, head_dim 128,
  short conv kernel 4, constant recurrent state, no per-token KV.
- 11 `deepseek_sparse_attention` layers (layers 3, 7, ..., 43): MLA with
  `kv_lora_rank` 512, `qk_nope_head_dim` 256, `v_head_dim` 256,
  `qk_rope_head_dim` 0 (`mla_use_nope`), `index_topk` 2048, 32 indexer heads,
  `index_head_dim` 128, `index_kpool` 4, pooled indexer with tail selection.
- mHC: `hc_mult` 4, 20 Sinkhorn iterations (the DSV4 engine already has HC).
- 1 nextn / MTP layer (layer 45), with its own 288 experts and DSA indexer
  (`index_share_for_mtp_iteration` true). No 3-stage DSpark.
- Vision: 24 blocks, 1024 hidden, patch 14, 448 px, temporal patch 2,
  spatial merge 2, out 4096.
- Vocab 154,880; context 1,048,576; GLM tokenizer and chat template
  (`reasoning_effort` low/high/max, `clear_thinking`), not the DSV4 text layer.

Parameter split used in the math:

- routed experts: 43 MoE layers (42 + MTP) x 288 experts x 3 matrices x 4096 x 2048
  = **311.65B** parameters
- everything else: **9.67B** parameters (attention ~6.1B estimated, shared experts
  1.06B, dense MLP 0.45B, embed/head 1.27B, vision ~0.53B, HC/norms/conv the rest)

## How the weight numbers were produced

The public EXL3 branches were read byte-exactly from their safetensors headers, not
from the model card:

1. List the branch's shard files with the HF tree API.
2. For each file, read the 8-byte little-endian header length with an HTTP range
   request, then range-read exactly that many header bytes.
3. Sum `data_offsets[1] - data_offsets[0]` per tensor and classify by tensor name:
   `.mlp.experts.` -> routed experts; `.mlp.shared_experts.` -> shared experts;
   `.mlp.` -> dense MLP; `.self_attn.` -> attention; `embed_tokens` -> embed;
   `lm_head` -> head; `hc_` -> hyper-connections; `norm` -> norms;
   `model.visual.*` -> vision; `layers.45.*` -> MTP layer.
4. The sums agree with the tree file sizes to within ~0.1 GB (headers, index,
   configs, and `kpool_aux.safetensors` are excluded).

Branches read: `4.05bpw`, `3.05bpw`, `2.05bpw`. Each includes the vision tower
and the MTP layer (the MTP experts are inside the routed-expert number; the separate
`mtp.safetensors` is part of the branch).

## Weight bytes and effective precision

Weights only, decimal GB:

| | routed experts | attention | embed | shared | vision | head | dense MLP + HC + MTP dense + norms | **total** | eff. bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **EXL3 4.05bpw** | 156.28 | 4.94 | 1.27 | 0.81 | 0.60 | 0.48 | 0.66 | **165.05** | 4.11 |
| **EXL3 3.05bpw** | 117.33 | 4.23 | 1.27 | 0.68 | 0.60 | 0.48 | 0.59 | **125.16** | 3.12 |
| **EXL3 2.05bpw** | 78.37 | 3.51 | 1.27 | 0.54 | 0.53 | 0.40 | 0.51 | **85.13** | 2.12 |
| native FP8 (approx. groups) | 311.66 | ~12.0 (BF16) | 1.27 | 1.06 | ~1.0 | 1.27 | ~0.6 | **328.25** | 8.18 |
| MXFP4 re-quant + 4.05 non-experts | 165.56 | 4.94 | 1.27 | 0.81 | 0.60 | 0.48 | 0.66 | **174.32** | 4.34 |

BF16 is 321.3B x 2 bytes = 642.6 GB; excluded. The native FP8 group sizes are
approximate because the checkpoint mixes BF16 (attention, embed/head, vision) and
FP8 (experts, shared, dense); its total comes from the parameter/dtype counts.

Effective bits by group (from the same bytes and parameter counts):

| branch | experts | non-experts | attention (approx.) |
|---|---:|---:|---:|
| 4.05 | 4.01 | 7.25 | 6.5 |
| 3.05 | 3.01 | 6.48 | 5.5 |
| 2.05 | 2.01 | 5.59 | 4.6 |

The branch's nominal bits apply to the experts. Varying `head_bits` is 6 for all
three; embed and vision are BF16; shared experts and attention stay higher
(5.1-6.5 bits). This is why 3.05 is the direct GLM analogue of DSV4 3.04.

Sizes that matter for download/space:

- 4.05 branch: 165.15 GB on the HF tree (19 shards + `mtp.safetensors`).
- 3.05 branch: 125.26 GB (15 shards + `mtp.safetensors` + `kpool_aux.safetensors`).
- 2.05 branch: 85.23 GB (12 shards).
- native FP8: 62 shards x ~5.36 GB = ~332 GB; weight payload 328.25 GB.
- The `sc_*` variants are 4.10/3.10/2.60/2.10 -> 167.81 / 128.03 / 109.00 / 88.28 GB.
  They use exllamav3 1.6.x and `out_scales: auto`; the "sc" meaning is undocumented,
  so they are not used here.

## KV and per-sequence state

Per token:

| component | values | bytes at fp8/fp4 | bytes at bf16 |
|---|---:|---:|---:|
| 11 DSA layers, latent KV (512 each) | 5,632 | 5,632 | 11,264 |
| 11 indexers (128 key + 128 gate + 1 valid) | 2,827 | 1,414 (fp4) / 2,827 (fp8) | 5,654 |
| **total per token** | | **7,046 (fp4 idx)** / **8,459 (fp8 idx)** | **16,918** |

Per sequence, KDA state:

- recurrent state: 34 layers x 64 heads x 128 x 128 x 4 B (fp32) = **142.6 MB/seq**
- conv state: 34 x 24,576 channels x 3 taps x 4 B ~= **10.0 MB/seq**
- **~152.6 MB/seq**

Totals:

| pool | fp8 KV + fp4 indexer | fp8 KV + fp8 indexer | bf16 KV |
|---|---:|---:|---:|
| 1,572,864 tokens (served geometry), 8 seq | 11.08 + 1.22 = **12.30 GB** | 13.31 + 1.22 = **14.53 GB** | 26.61 + 1.22 = **27.83 GB** |
| 2,097,152 tokens (4 x 512k), 4 seq | 14.77 + 0.61 = **15.38 GB** | 17.74 + 0.61 = **18.35 GB** | 35.48 + 0.61 = **36.09 GB** |
| 4,194,304 tokens (4 x 1M), 4 seq | 29.55 + 0.61 = **30.16 GB** | 35.48 + 0.61 = **36.09 GB** | 70.96 + 0.61 = **71.57 GB** |

DSV4's served cache is 9.41 GB at the same 1,572,864-token pool (~5.98 KB/token).
GLM is +2.9 to +5.1 GB with fp8/fp4 and +18.4 GB with bf16. DeepSeek is only
slightly more efficient; the gap is the full-resolution DSA latent plus the
per-token indexer state.

**Important for the cache manager:** the KDA state is unpaged per-sequence state.
At 152.6 MB/seq it is ~22x the DSV4 per-slot state (~7 MB). Prefix-cache snapshots
can no longer be ~7 MB each: 64 snapshots at 152.6 MB would be 9.8 GB of pinned
host memory. Either keep far fewer GLM snapshots, store them at fewer boundaries,
or compress the state (bf16 halves it but changes bits). This is a design decision
for the port, not a quant decision.

## Fit at the served geometry

Capacity 193.53 GB; scratch assumed to stay 6.82 GB (GLM's could be larger; treat
as an upper bound on headroom). fp8 main KV + fp4 indexer unless stated.
These are planner-equivalent projections: the engine cannot load GLM yet. GLM's
step scratch is the largest unknown; if it is materially above 6.82 GB, the 4.05
and MXFP4 rows lose headroom first.

| build | weights | KV + state | scratch | total | headroom |
|---|---:|---:|---:|---:|---:|
| served DSV4 native | 168.22 | 9.41 | 6.82 | 184.45 | **+9.08** |
| **GLM 4.05** | 165.05 | 12.30 | 6.82 | 184.17 | **+9.36** |
| **GLM 3.05** | 125.16 | 12.30 | 6.82 | 144.28 | **+49.25** |
| GLM 2.05 | 85.13 | 12.30 | 6.82 | 104.25 | +89.28 |
| GLM 4.05, bf16 KV | 165.05 | 27.83 | 6.82 | 199.70 | **-6.17** |
| GLM native FP8 | 328.25 | 12.30 | 6.82 | 347.37 | **-153.84** |
| GLM MXFP4 + 4.05 non-experts | 174.32 | 12.30 | 6.82 | 193.44 | +0.09 (on the edge) |

Other pools:

| build | 4 x 512k (2M pool) | 4 x 1M (4M pool) |
|---|---:|---:|
| GLM 4.05 | 165.05+15.38+6.82 = 187.25 (**+6.28**) | 165.05+30.16+6.82 = 202.03 (**-8.50**) |
| GLM 3.05 | 125.16+15.38+6.82 = 147.36 (**+46.17**) | 125.16+30.16+6.82 = 162.14 (**+31.39**) |
| GLM 2.05 | 85.13+15.38+6.82 = 107.33 (+86.20) | 85.13+30.16+6.82 = 122.11 (+71.42) |

Conclusions:

- **4.05 is the drop-in match for the native DSV4 currently served**: 165.05 GB
  vs 168.22 GB, total 184.2 vs 184.5 GB, ~9.4 vs ~9.1 GB headroom. It fits only
  with fp8 KV; bf16 KV does not fit the served pool.
- **3.05 is the drop-in match for the DSV4 EXL3 3.04 build** (125.16 GB vs ~125.3 GB
  plan footprint) and is the only one of the two that supports 4 x 1M contexts.
- **2.05** is unnecessary headroom unless its quality is proven.

## Hot experts and decode bytes

Cards have ~36.7 GB for experts at the served geometry after 4.05-class non-experts
(8.17 GB), KV+state (12.30 GB), and scratch (6.82 GB):

| | expert size | experts on cards | share of 12,384 | APU cold (1.57M) |
|---|---:|---:|---:|---:|
| 4.05 | 12.62 MB | ~2,910 | 23.5% | 119.6 GB |
| 3.05 | 9.47 MB | ~3,875 | 31.3% | 80.6 GB |
| 2.05 | 6.33 MB | ~5,800 | 46.8% | 41.7 GB |
| native FP8 | 25.17 MB | ~1,460 | 11.8% | -- |
| MXFP4 | 13.37 MB | ~2,745 | 22.2% | -- |

At 4 x 1M the card expert budget shrinks to ~18.9 GB (KV is 30.16 GB), so the hot
share falls to ~12.1% at 4.05 and ~16.1% at 3.05. Long context is therefore more
APU-bound; this is inherent to storing full-resolution DSA KV.

Decode weight reads per token (per class, resident):

| build | routed | shared | attention | head + dense + HC | total/token |
|---|---:|---:|---:|---:|---:|
| 4.05 | 4.28 | 0.81 | 4.94 | 1.00 | **11.03 GB** |
| 3.05 | 3.22 | 0.68 | 4.23 | 0.95 | **9.08 GB** |
| 2.05 | 2.17 | 0.54 | 3.51 | 0.81 | **7.03 GB** |
| served DSV4 native | ~3.7 | ~1.1 | -- | -- | **~6.9 GB** |

3.05 is within ~30% of the served DSV4's per-token read; 4.05 is ~60% more.
Rough expectations before kernel tuning: 3.05 ~20-30 tok/s, 4.05 ~16-22 tok/s.
The single MTP layer can draft tokens but is not a 3-stage DSpark, so do not assume
the DSV4 draft acceptance profile.

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

**Native FP8 (328.25 GB):**

- Resident expert capacity with EXL3 4.05 non-experts:
  hot ~36.7 GB on cards + 128.9 GB on the APU = ~165.6 GB of 311.65 GB.
  Cold ~146 GB.
- Prefill ceiling at 4096-token chunks: `4096 * 5.8 / 146 = 163 tok/s`.
  At 1024-token chunks: `1024 * 5.8 / 146 = 41 tok/s`.
- Decode: active routed FP8 = 8.46 GB/token. Cold routing fraction ~0.3-0.4
  (the hot set captures most of the mass): 2.5-3.4 GB/token -> 438-583 ms/token
  -> **1.7-2.3 tok/s**. Uniform 48% gives 1.4 tok/s.
- With native BF16 attention/non-experts instead of the EXL3 dense the cold set is
  ~154 GB and the prefill ceiling ~154 tok/s; conclusions unchanged.

This cannot meet the project targets (500 tok/s prefill, 20 tok/s decode). It is a
research baseline, not a serving mode.

**MXFP4 re-quant + EXL3 4.05 non-experts (174.32 GB):**

- Resident capacity ~165.6 GB of 165.56 GB experts: essentially all experts fit.
  At the full 1,572,864-token pool the total is 193.44 GB vs 193.53 GB capacity
  (+0.09 GB, too tight to rely on once GLM's scratch is known). At 1M tokens it is
  189.75 GB (+3.8 GB).
- If a few GB are cold, the SSD cost is small: active routed MXFP4 is 4.49 GB/token,
  and a 2-5% cold fraction is 0.09-0.22 GB/token -> 15-38 ms/token of SSD time.
  Prefill cold bytes per 4096 chunk are ~GB, so prefill is not SSD-bound.
- Requires building an MXFP4 quantizer for GLM's FP8 experts; quality vs the
  calibrated EXL3 4.05 is unmeasured. Not needed for the main path because 3.05
  already fits with ~49 GB spare.

Space is no longer an issue: `/scratch` has 1.8T free (was 230G before cleanup).
Native FP8 (332 GB of shards) plus 3.05 (125 GB) and 4.05 (165 GB) all fit.

## Implementation notes for the next agent

The quant choice is settled; the port is the large part. GLM is not a DSV4 variant:

1. **KDA layers (34).** Chunked/recurrent Kimi delta rule with q/k/v projections
   (64 heads x 128), short causal conv (kernel 4), forget gate, per-head beta,
   gated output norm, fp32 recurrent state of 64x128x128 per layer per sequence.
   New kernels and a new state layout; no existing DSV4 kernel covers this.
2. **DSA layers (11).** MLA without rope (q_lora 1536, kv_lora 512, nope 256,
   v 256), top-2048 selection, pooled indexer (kpool 4, 32 heads x 128,
   `index_kpool_compress_ape/gate`, tail selection). `indexer_types` is all "full"
   here, so no cross-layer top-k sharing is needed initially, but keep the hook.
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
7. **Cache manager.** DSA KV pages: 1 token per row x 512, fp8 + block scales;
   indexer state per token (pooled at read time), fp4 or fp8; KDA state snapshots
   as discussed (152.6 MB/seq, not 7 MB). Prefix caching needs a new snapshot
   budget/strategy.
8. **Planner.** 43 MoE layers (42 + MTP) x 288 experts; expert bytes per format
   12.62 / 9.47 / 6.33 / 25.17 / 13.37 MB for EXL3 4.05 / 3.05 / 2.05 / FP8 / MXFP4;
   vision 0.60 GB on the APU; no DSpark experts. The planning code is otherwise
   format-agnostic.
9. **Serve geometry.** For parity with the served DSV4 use
   `--kv-tokens 1572864 --max-model-len 786432 --max-num-seqs 8`.
   For the DESIGN target with 3.05 use `--kv-tokens 2097152 --max-model-len 524288`
   (4 x 512k) or `--kv-tokens 4194304 --max-model-len 1048576` (4 x 1M).
   Both fit at 3.05; only the first fits at 4.05.
10. **Verification.** There is no DSV4-style oracle for GLM in the tree. Use the
    HF transformers `glm5_next` implementation running the native FP8 checkpoint
    as the oracle, layer-streamed on one card, the same way `tools/ref/oracle`
    does for DSV4. Then compare per-layer and end-to-end KL/top-1.

## Open questions / next measurements

- **Quality:** KL and top-1 of 3.05 and 4.05 against native FP8 on the same
  1000+ position prompt used for the DSV4 oracle. The decision rule: if 3.05 is
  in the same class as the DSV4 3.04 result (KL ~1e-2, top-1 within ~1% of the
  reference's own reordering noise), take 3.05.
- **Scratch:** GLM's step scratch for KDA + 288-expert MoE + pooled indexer is not
  measured. If it exceeds 6.82 GB, the 4.05 headroom shrinks and MXFP4 stops fitting.
- **Mixed checkpoint packaging:** merge 3.05 experts with 4.05 non-experts on disk,
  or extend the loader to take two EXL3 sources. The current `--model` (native)
  plus `--exl3` (experts) split is not the right shape for EXL3 dense + EXL3 experts.
- **Prefix snapshots:** 152.6 MB per KDA state snapshot forces a different policy
  (fewer boundaries, host-resident, or compressed). Measure the cost of restoring
  a bf16 state versus recomputing.
- **`sc_*` branches:** possibly better quantizer (exllamav3 1.6.x, auto out-scales),
  undocumented calibration. Revisit only if 3.05/4.05 quality disappoints.
- **MTP yield:** 1 nextn layer, unknown acceptance on this machine. Measure before
  committing serve defaults to speculative decoding.

## Reproduction

Plan on son-of-anton:

```sh
cd /scratch/unholy
./build/unholy plan --model /scratch/dsv4-ve --profile data/expert_profile_unholy_mixed.txt \
  --kv-tokens 1572864 --max-model-len 786432 --max-num-seqs 8 --max-batch-tokens 4096 \
  --prefix-snapshots 64 --spec-drafts 5
```

The `--exl3` variant adds `--exl3 /scratch/models/dsv4-ve-exl3-3.04bpw`.

Safetensors header scan (per branch): read `curl -r 0-7` for the header length,
then `curl -r 8-(8+len-1)` for the JSON header, and sum
`data_offsets[1]-data_offsets[0]` grouped by tensor-name class. The exact
classification rules are in "How the weight numbers were produced".

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
  `Glm5NextTextIndexer` for the pooled indexer cache)
- Model blog: https://z.ai/blog/glm-5.3-flash
- Served DSV4 plan: `unholy plan` on son-of-anton, 2026-10-22.

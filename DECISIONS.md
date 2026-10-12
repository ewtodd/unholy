# DECISIONS

Dated decisions with the evidence behind them. Newest last.

## 2026-10-11 — restart as `unholy`

The owner's requirements (DESIGN.md) rule out the three previous paths:

- `/scratch/dsv4-engine` (custom C++/HIP): 745 tok/s prefill at 32k but 3.2 tok/s decode,
  one sequence, a host-serialized scheduler (`block.hip` alone had 92 `hipDeviceSynchronize`).
- llama.cpp `ds4-hybrid` (sixvolts/llama-halo-hybrid, local branch): 927 tok/s at 32k,
  761 at 100k, 20.2 tok/s decode, but llama.cpp slots, not continuous batching or paged
  state; concurrent prefill is slower than serial (605/843 vs 949 aggregate).
- vllm.cpp: builds on ROCm, but its DS4 device forward is a `VT_CHECK(false)`; its placement
  is CPU/GPU only and it fit a 380-token KV budget.

So: a new tree, vLLM-shaped from the first commit (paged state, scheduler, plan-driven
executor), taking kernels and lessons from all of them plus gufo, Strata and exllamav3
(DESIGN.md "Donors").

## 2026-10-11 — model: DeepSeek-V4-Flash-Vision-Exp, not V4.1-Flash

`deepseek-ai/DeepSeek-V4.1-Flash` (2026-10-01) has `text_config`: 40 layers, hidden 5120,
384 routed experts x inter 2304, q_lora 1280, index_n_heads 32, compress_ratios 2/1 (a
different compression scheme). Routed parameters alone: 40 x 384 x 3 x 5120 x 2304 =
543.6B -> ~204 GB at 3 bpw, more than the 64 GB of VRAM plus ~115 GB of usable APU memory.
Published EXL3 quants of it target 2x RTX PRO 6000 (192 GB). Out of scope.

Target: V4-Flash-Vision-Exp (same 284B architecture as V4-Flash / 0731, plus the vision
tower), because it is already local (`/scratch/dsv4-ve`, native) and turboderp publishes EXL3
quants of exactly it (`turboderp/DeepSeek-V4-Flash-Vision-Exp-exl3`, branches 2.04-3.04 bpw).
The 3.04 bpw branch (118.4 GB, exllamav3 1.4.8, `mul1` codebook; `bits` 3.04, `head_bits` 5,
`vision_bits` 5, `mtp_bits` 3; attention projections stored at 5 bpw) is at
`/scratch/models/dsv4-ve-exl3-3.04bpw`.

## 2026-10-11 — the probe: what moves data on this machine, and what does not

`unholy-probe` (tools/probe) on the current layout, ROCm 10.1.0 TheRock SDK; full record in
`docs/probe/son-of-anton-2026-10-11.json`. Device order: 0, 1 = R9700 (gfx1201, 32 WGPs),
2 = Strix Halo iGPU (gfx1151, 20 WGPs, 124 GiB).

Links. Each card sits behind its own Gen4 x4 root port (00:02.5, 00:03.2); the R9700's
internal switch reports 32 GT/s x16 on the endpoint, and idle root ports show 2.5 GT/s
(ASPM), which is why the planner uses the minimum of every hop's *maximum* link, then
measures.

| path | measured | verified |
|---|---|---|
| host -> card / card -> host (pinned) | 7.16 / 6.96 GB/s | yes |
| both cards at once, h2d / d2h aggregate | 14.4 / 14.0 GB/s (separate root ports today) | yes |
| host -> APU / APU -> host | 46.1 / 68.3 GB/s | yes |
| card local read | 595 / 535 GB/s (card 0 / 1) | — |
| APU local read / APU read of pinned host | 237 / 234 GB/s | — |
| card read of pinned host (zero-copy) | 7.2 GB/s | — |
| card <- card, kernel pull (reader on the destination) | 7.1 GB/s | yes |
| card <- APU memory, kernel pull | 7.2 GB/s | yes |
| card -> APU memory, kernel push | 7.2 GB/s | yes |
| host-staged pipeline (d2h + h2d, 16 MiB slots, event-ordered) | 6.5-7.1 GB/s | yes |
| `hipMemcpyPeerAsync`, any pair | 5.6-7.1 GB/s | yes |

Broken — never used by the engine:

- **APU reading card memory** (kernel pull into the APU): every word wrong, at an
  impossible 23 GB/s. The iGPU's peer mapping of the R9700 BAR does not reach the card.
- **Kernel push into a card's coarse-grained memory** (from the other card or the APU):
  exactly ~30 MiB of the 256 MiB destination reads stale after the writer's stream
  completes. Incoming PCIe writes do not invalidate the destination's caches for a
  following kernel. Pull (the reader issues the loads) is correct.
- **APU writing a flag into card fine-grained memory**: never observed by the card.
- **`hipMemcpyPeer` (sync)** returns before the copy runs ("13-62 TB/s"); the data was
  correct here only because the check kernel queued behind it.

Transport rule that follows: the reader pulls, or the card pushes into APU/host memory; the
APU only ever touches its own memory and pinned host memory. Every card <-> APU exchange
uses APU-local or pinned host buffers as the rendezvous.

Signalling round trip (A signals B, B signals back), per mechanism:

| mechanism | card<->card | card<->APU |
|---|---|---|
| host `hipStreamSynchronize` each way | 64 us | 41-51 us |
| `hipStreamWaitEvent` cross-device, pre-enqueued | 22 us | 18 us |
| `hipStreamWaitValue32`/`WriteValue32` on coherent host flags | 14 us | 12 us |
| persistent kernels spinning on coherent host flags (device-timed) | 3.2 us | 2.3-2.6 us |
| persistent kernels on peer fine-grained flags | 2.4 us | fails (APU->card write) |

The decode critical path has 43 card<->APU hops per token: at 2.5 us that is ~0.1 ms/token,
at 12 us ~0.5 ms. The APU hop is not the decode bottleneck as long as the host stays off the
path (host sync would be ~2 ms/token). The two stream-level numbers include host enqueue cost
and are upper bounds.

## 2026-10-11 — EXL3 decoded exactly; first weight-space error numbers

`src/model/exl3.cpp` decodes EXL3 linears on the CPU (format notes in `exl3.hpp`): tail-biting
16-bit trellis windows over an MSB-first bitstream of little-endian uint32 words, the `mul1`
codebook (`x * 0x83DCD12D`, byte sum + 0x6400 as fp16, `fma(h, 1/147.7, -10.39)`), the
tensor-core tile permutation, and W = diag(suh) H128 W_inner H128 diag(svh). Checked against
exllamav3's own `get_inner_weight_tensor()` (built from source on son-of-anton against the vLLM
stack's ROCm torch, `tools/ref/build_exllamav3.sh`): **0 mismatches** over 21M weights
(experts.0.w1 / experts.17.w2 at K=3, attn.wq_a at K=5); dense weights within exllamav3's per-step
fp16 rounding (rel 4.1e-4).

Weight-space error of the turboderp 3.04 bpw checkpoint against the native weights
(`unholy-ckpt compare`):

| tensor | K | rel L2 | cosine |
|---|---|---|---|
| routed experts w1/w2/w3 (layers 0, 20, 42) vs MXFP4 | 3 | 0.144-0.159 | 0.9874-0.9897 |
| shared expert w1 vs fp8 | 5 | 0.038 | 0.9993 |
| attention wq_a / wq_b / wkv / wo_b vs fp8 | 5 | 0.038-0.041 | 0.9992-0.9993 |

The native MXFP4 experts are the trained weights (QAT), so ~15% expert weight error is pure loss.
Output KL decides, but this already argues for (a) native fp8 attention (lossless, and the fast
path on gfx1201) and (b) measuring MXFP4-hot / EXL3-cold mixes and a 4 bpw EXL3 tier.

## 2026-10-11 — the oracle: DeepSeek's own model.py, layer-streamed

`tools/ref/oracle/run.py` runs the checkpoint's own `inference/model.py` (MIT, shipped in
`/scratch/dsv4-ve/inference`) **unmodified** on one R9700 through the vLLM stack's ROCm torch,
one `Block` at a time (build, load weights, push every position through, free), teacher-forced:
the prompt as one prefill at `start_pos` 0, then the continuation one token at a time, which
exercises the decode-path state (window ring, compressor `kv_state`/`score_state`, indexer).
`tools/ref/oracle/kernel.py` replaces the tilelang kernels with torch code that keeps their
arithmetic: power-of-two scales by the same ceil(log2) bit rule, amax floors, fp32 per-block
(128 fp8 / 32 fp4) partial sums scaled then accumulated, 64-key online-softmax blocks with bf16
probabilities, the sink added last, the 20-iteration Sinkhorn.

Validation: on `ids_code32768.bin[:1024]` the oracle's next-token argmax is **2606** (logit
33.141) and the following argmaxes are `[223, 18, 11, 4054, ...]` — the same five tokens
dsv4-engine and gufo (antirez's quant) agreed on. A full 43-layer run (1024 prefill + 8 decode)
takes ~2.5 min and dumps every layer's HC residual and the full logits.

What reading the reference changed in the design (DESIGN.md updated):
- the checkpoint's `mtp.*` layers are a **DSpark** block drafter (block 5, target layers
  40-42, markov head rank 256, confidence head), not classic MTP;
- the reference is QAT'd at fp8 activations (128-blocks) for every fp8/fp4 linear, fp8-simulated
  KV (64-blocks, nope dims only) and fp4-simulated indexer q/k after a 128-point Hadamard — so
  storing KV as fp8 and indexer keys as fp4 loses nothing against the reference (~1.8 GB per
  512k slot instead of 3.6 GB);
- `wo_a` is dequantized to bf16 and run as a bf16 grouped einsum; compressor `wkv`/`wgate`, the
  router and the LM head run in fp32.

## 2026-10-11 — the reference path is correct (KL at the noise floor)

`src/model/dsv4.hip` + `src/kernels/ref_*.hip`: the whole forward in our own C++/HIP, following
model.py's rounding points, generalised to chunked prefill (window ring, compressor projection rings,
paged compressed rows, page tables). `unholy-ref-run` streams all 43 layers on one R9700 (1024-token
prefill + 8 teacher-forced decode steps, 63 s including loading) and compares logits with the oracle:

| comparison (1032 positions, code prompt) | mean KL | median KL | p99 | top-1 agree |
|---|---|---|---|---|
| oracle vs oracle with GEMM K-blocks summed in reverse (`ORACLE_PERTURB=1`) | 9.9e-3 | 1.1e-3 | 8.8e-2 | 96.2% |
| oracle vs unholy reference path | 9.6e-3 | 1.0e-3 | 9.9e-2 | 95.7% |

Every argmax after the prompt matches (2606, 223, 18, 11, 4054, 223, 18, 4658, 1313). The engine is
as close to the oracle as the oracle is to itself under an equally valid fp32 summation order: the
fp8-activation QAT design amplifies 1-ulp differences (see the per-layer bisection in STATUS.md), so
this ~1e-2 mean KL / ~96% top-1 is the floor for any re-implementation, and the reference for judging
the fast path and the quantization study is "within the noise floor", not "bit-identical".

Loading: mmap + hipMemcpy from pageable memory faulted the 168 GB checkpoint in a page at a time once
it no longer fit the page cache (20+ min per pass). `src/model/loader.hip`: O_DIRECT reads of
4 KiB-aligned ranges straight into pinned staging buffers, double-buffered against async H2D copies,
a layer's ~1500 tensors sorted by offset and coalesced into chunk-sized reads: 0.64 s per 3.6 GB layer
(~5.6 GB/s), ~30 s for the whole checkpoint.

## 2026-10-11 — first quant number: EXL3 3.04 experts

`src/kernels/ref_exl3.hip`: reference EXL3 linear (H(H(x*suh) W_inner)*svh in fp32, W_inner decoded
from the trellis in the GEMM loader); `load_layer(..., exl3=...)` takes the routed experts from the
EXL3 checkpoint, everything else native (fp8 attention/shared experts). The expert inputs keep the
fp8 fake quantization the experts were trained with (A8), so only the weights change.

`unholy-ref-run --exl3 /scratch/models/dsv4-ve-exl3-3.04bpw` vs the native-MXFP4 oracle, code prompt,
1032 positions:

| | mean KL | median KL | p99 | top-1 agree |
|---|---|---|---|---|
| noise floor (oracle vs reordered oracle) | 9.9e-3 | 1.1e-3 | 8.8e-2 | 96.2% |
| native experts (reference path) | 9.6e-3 | 1.0e-3 | 9.9e-2 | 95.7% |
| **EXL3 3.04 experts** | **2.7e-2** | 3.4e-3 | 3.1e-1 | 92.7% |

All 9 post-prompt argmaxes still match. 3-bit experts cost ~1.7e-2 nats over the floor on this
prompt. One prompt type only; the quant study needs a mixed eval set (prose, chat, math, code, long
context) and per-expert mixes (MXFP4 for the most-used / most-sensitive experts).

## 2026-10-11 — batched, paged forward; chunk- and batch-invariant

`src/model/forward_ref.{hpp,hip}` replaces the single-sequence forward: a batch is a list of chunks
(slot, s0, n, t0); HC mixing, norms, projections, wo_a/wo_b and the MoE run over all T tokens,
compressors / indexer / attention / window ring per chunk. Compressed rows (CSA 64, HCA 2, indexer 64
rows per 256-token page) live in per-layer pools addressed through one block table per slot (physical
page ids shared by all layers on a device, so pages are allocated once per position range). Window
rings and compressor rings are per-slot arrays. The MoE writes every (token, expert) pair to its own
output row and `moe_combine` sums each token's pairs in ascending expert order, so hot and cold
experts can later run on different devices without changing the reference accumulation order.

Checks: the oracle comparison is unchanged to every printed digit; prefill in 300-token chunks (not
aligned to the 4/128 compression blocks) gives bit-identical output to one 1024-token chunk;
`unholy-batch-test` runs three sequences (600/523/333 tokens) interleaved in mixed batches (chunks of
1/37/128/211 tokens, shuffled slots, shuffled physical pages) and every element matches the solo runs
on layers 0 (window), 2 (CSA) and 3 (HCA).

## 2026-10-11 — the model resident across both cards and the APU

`src/engine/plan.cpp` sizes everything from the checkpoint and the real allocation functions
(`dense_layer_bytes`, `expert_bytes`, `layer_cache_bytes`, `scratch_bytes`, `hipMemGetInfo`): layers
split to balance dense + cache bytes (0-21 / 22-42), each card's remaining VRAM filled greedily with
that card's most-routed experts (usage from `data/expert_profile_dsv4engine_code32k.txt`, dsv4-engine's
32k-token code calibration), every other expert on the APU. With EXL3 3.04 experts, 1 slot x 2k:
card 0 2907 hot experts (27.5 GB), card 1 2861 (27.1 GB), APU 49.6 GB cold, **95% of routing predicted
on the cards**.

`src/engine/resident.hip`: per-card weights/caches/scratch, the APU's experts behind
`RemoteExperts` (inputs and outputs in pinned host memory: the APU reads and writes it zero-copy, the
cards DMA), residual handoff card 0 -> card 1 through pinned memory, head on card 1. Weights come from
per-device arenas (256 MiB slabs): one hipMalloc per tensor rounded each 3 MiB trellis up to 4 MiB and
ran the cards out of memory. The uploader's events are recreated when the caller switches devices.

`unholy-resident-test` (load 37.7 s; code prompt, 1024 prefill + 8 teacher-forced decode steps):
**all 133,416,960 logits bit-identical to the single-card streaming run**, KL vs the oracle 2.667e-2
(the EXL3 number). Reference-kernel speed, the baseline the fast path replaces: prefill 1024 tokens
27.5 s (37 tok/s), decode ~1.0 s/token.

## 2026-10-11 — grouped EXL3 MoE kernels (fast path, first piece)

The reference profile (rocprofv3, resident run): 41.3 of 50.8 GPU-seconds (81%) in the reference
EXL3 GEMM (per-element trellis decode, 35664 launches), sparse_attn 3.7 s, W8A8 GEMMs 2.8 s.

`src/kernels/exl3_moe.hip`, one launch per projection for every resident expert of a layer:
`exl3_xform_in` (A8 fake quant + suh + 128-point Hadamard -> fp16, per (row, 128-segment)),
`exl3_grouped_gemm` (tile list of <= 64 rows of one expert x 128 columns; the K-slab's trellis words
into LDS, decoded to fp16 [n][k], fp16 WMMA with fp32 accumulators, gfx12 and gfx11 fragment
layouts; epilogue: output Hadamard over the 128 columns in LDS * svh -> bf16), `exl3_swiglu_xform`
(clamped SwiGLU * routing weight, bf16 round, A8 fake quant, w2's suh + Hadamard -> fp16). Pointer
tables per layer and device ([w1, w3, w2] x [trellis, suh, svh] x E, null where the expert lives
elsewhere) are built at load; the same kernels run on the APU (gfx1151) for the cold experts.
Not adopted: exllamav3's own fused MoE (its RDNA kernels are excellent but bound to its Python tier
orchestration); its decode math and the tensor-core permutation are what these follow.

Results: one layer (3, HCA) vs the oracle is identical to the reference EXL3 path to 3-4 digits
(prefill rel 3.592e-2 both; the difference is the fp16 rounding of the transformed input); resident
model, code prompt: KL 2.52e-2 (reference EXL3 path 2.67e-2), prefill 1024 tokens 27.5 -> 8.0 s,
decode 1.0 -> 0.47 s/token, load 38 -> 22 s.

## 2026-10-11 — WMMA dense GEMMs

`src/kernels/gemm_wmma.hip`: fp8 x fp8 WMMA (gfx12) for every W8A8 linear, with the reference's
structure kept exactly (each 128-wide K block accumulates in a raw WMMA accumulator, folded into the
output with sa[m][kb] * sw[n/128][kb] at the block end); bf16 WMMA (gfx11 + gfx12) for the fp32
linears whose inputs are bf16-valued (compressor projections, router, head: products exact, only
summation order differs), the indexer weights and the grouped wo_a (fp8 x 128x128 scale dequantized
to bf16 in LDS, exact). Dispatch in forward_ref (`UNHOLY_REF_DENSE=1` keeps the references).
Layer 2 vs oracle unchanged (prefill rel 4.56e-3 vs 4.48e-3); resident EXL3 model: KL 2.64e-2,
prefill 1024 tokens 8.0 -> 6.67 s, decode 0.47 -> 0.24 s/token.

## 2026-10-11 — WMMA sparse attention

`src/kernels/attn_wmma.hip` (gfx12): one 256-thread block per query computes all 64 heads (MQA: one
KV row set for all heads, loaded once). Wave (hg, dh) holds heads 16hg.. and dims 256dh.. : Q slice as
bf16 WMMA A fragments in registers, output slice in 16 accumulators; keys 16 at a time staged in LDS
key-major (Q K^T) and dim-major (P V); partner waves sum partial scores through LDS; per-head online
softmax with bf16 P, sink in the denominator. Softmax blocks are 16 keys instead of the reference's
64 (bf16 rounding of P against a different running max: noise-level). Layers 2/3 vs oracle:
4.62e-3 / 7.79e-3 (reference attention 4.56e-3 / 7.67e-3). Resident EXL3 model: KL 2.59e-2, prefill
1024 tokens 6.67 -> 3.06 s (~335 tok/s), decode 0.23 s/token.

## 2026-10-11 — head on WMMA, contiguous remote pairs, token-tiled HC mixing

- LM head: bf16 WMMA with fp32 output on the bf16 normed state (the reference's fp32 linear, exact
  products).
- Pair rows are laid out local experts first, then remote (APU) experts, so the rows shipped to the
  APU and back are one contiguous block each way (was one copy per remote expert: ~71k tiny
  copyBuffer launches per run). The combine follows pidx, so the summation order is unchanged.
- `hc_pre_tiled` (src/kernels/hc_fast.hip): 16 tokens per block share each LDS-staged chunk of the
  [24, 16384] fp32 mixing matrix (the reference kernel re-read 1.5 MB per token).
Resident EXL3 model: KL 2.44e-2, prefill 1024 tokens 3.06 -> 2.68 s, decode 0.24 s/token.

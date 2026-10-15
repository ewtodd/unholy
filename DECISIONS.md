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

## 2026-10-11 — device-side MoE dispatch, APU behind stream flags

`src/kernels/moe_dispatch.hip`: count -> plan (pair offsets with resident experts first, remote0, the
<=64-row tile lists of both groups) -> scatter (each token's picks sorted by expert, atomic row
placement within an expert's block; row-independent experts and the pidx-ordered combine make the
placement irrelevant to the result) -> gather. The grouped EXL3 GEMM takes its tile count from
device memory (grid sized for the maximum). Remote rows, weights, experts, plan and tiles go to
pinned memory on the card's stream; `ApuExperts::enqueue` raises a sequence-numbered request flag
(hipStreamWriteValue32), the APU stream waits on it, runs the remote tiles, raises done; `join`
makes the card stream wait for done. The compressor's row positions are computed on the device. The
EXL3 fast path now has no host synchronization in a layer.

Single EXL3 layer on one card, 1024 prefill: 162 -> 75 ms; resident model: KL 2.438e-2 (same),
prefill 1024 tokens 2.68 -> 2.44 s, decode 0.24 -> 0.20 s/token (~4.4 ms per layer per token).

## 2026-10-11 — small-M (decode) kernels

The decode trace of one layer (T = 1, 49 launches, 4.4 ms) showed prefill-shaped kernels: the tiled
HC pre-mix ran one block over K = 16384 (575 us x2), the router one thread with a serial top-6
(543 us), the EXL3 GEMM 64-row tiles holding one row (333 us per projection), the WMMA GEMMs
4-block grids for small N. Added (`src/kernels/gemv.hip`, `hc_fast.hip`, `ref_moe.hip`,
`exl3_moe.hip`), selected by shape:
- skinny GEMMs for M <= 8: W8A8 (each 16-wide chunk's exact sum scaled by its 128-block scale
  product), bf16 -> bf16 / fp32, grouped fp8-weight wo_a; wave per column, lanes over K;
- HC pre-mix for T <= 32: 32-way split-K partial sums, then Sinkhorn + mix per token;
- router: a block per token, parallel scores, K argmax rounds (ties -> lower id, as the reference);
- EXL3 GEMV for batches of <= 8 tokens: block per (tile, 16-column block), K over 8 waves, each lane
  decoding its 8 trellis steps of a tile in registers (rows {r0, r0+1, r0+8, r0+9} x columns
  {c0, c0+8}), then the output Hadamard * svh per row.
Decode error vs the oracle unchanged (layer 3: 3.612e-2); per-layer decode 4.4 -> ~1.7 ms;
resident model decode 0.20 s -> 79 ms/token (12.7 tok/s raw), KL 2.438e-2.
Follow-up from the full decode-step trace (87 ms, 3702 launches): the EXL3 GEMV/out grids were sized
for the worst-case tile count (257) and launched ~50k empty blocks per call (now bounded by the
pair-row count), the HC finish ran Sinkhorn out of scratch memory (runtime-indexed arrays; now
compile-time hc = 4, registers), and the LM head used the 64-row WMMA tile at M = 1 (now the GEMV;
1.06 GB of bf16 head per token is its floor). Decode 79 -> 58 ms/token (~17 tok/s raw), KL unchanged.

## 2026-10-11 — serving text layer: tokenizer, chat format, output parser

Own implementations rather than embedding Python or the Rust `tokenizers` crate: the server is one
C++ binary.
- JSON (`src/text/json.cpp`): ordered, numbers kept as their source text, `dump_py()` = Python's
  `json.dumps(ensure_ascii=False)` (", "/": " separators, ints as ints, floats as `repr`). The chat
  format embeds tool schemas and arguments through `json.dumps`, so this has to be exact.
- Tokenizer (`src/text/tokenizer.cpp`): HF's pipeline for this tokenizer.json — added tokens cut out
  first (special + non-normalized in one leftmost-longest pass, normalized ones in a second), the three
  Isolated Split regexes each applied to the pieces of the previous one, byte-level alphabet, BPE by
  merge rank with HF's queue order. PCRE2 (UTF+UCP, JIT) stands in for Oniguruma; the one semantic
  difference found is `\s`: PCRE2's includes U+180E, Oniguruma's is the White_Space property, so
  `\s`/`\S` are rewritten to the explicit set at load. Against HF on 91k strings (adversarial
  whitespace/digits/CJK/emoji/special-token fragments + real source files): identical except 2 strings
  containing code points newly assigned in Unicode 17 (PCRE2 10.48 has newer tables than HF's
  Oniguruma; not worth emulating). Decode (incl. a streaming detokenizer holding back incomplete UTF-8)
  identical on 6k sequences. 5.3M tok/s vs HF's 0.76M on the same 3 MB text (a 512k prompt in ~0.1 s).
- Chat format (`src/text/chat.cpp`): a port of the checkpoint's `encoding_dsv4.py` text path, plus
  vLLM's two deliberate changes: a system message that is last or followed by an assistant turn opens
  a generation turn, and object-valued tool-call arguments render as parameters. Request semantics as
  vLLM's DSV4 tokenizer wrapper: tools become a leading system message, thinking defaults on,
  `reasoning_effort` defaults to high when thinking (none -> chat mode; low/minimal/medium -> low).
  Golden files 4/4, 20k randomized conversations byte-identical to the reference (cases with the two
  deliberate differences filtered), 400 OpenAI requests identical to vLLM's `apply_chat_template` in
  text and token ids.
- Output parser: vLLM's DSV4 state machine (reasoning until `</think>`, content, DSML tool blocks
  anywhere, arguments via `_dsml_arg_converter` + wrapper unwrapping), streaming with held-back
  partial terminals; the "\n\n" before a tool block is dropped from content as the reference does.
  Identical to `parse_message_from_completion_text` on 24k reference-rendered completions fed in
  chunks of 1..13 bytes and whole.

## 2026-10-11 — engine loop, scheduler, OpenAI server: first served tokens

- `StepModel` (`src/engine/step.hpp`) is the scheduler's whole view of a model (one batched step over
  chunks + block tables), so the scheduler is tested on the CPU: `unholy-sched-test` runs a fake model
  that stores tokens in its own paged pool through the block tables it is given and derives one-hot
  logits from the whole sequence read back through them; greedy outputs must equal a standalone
  simulation under random arrivals, aborts, tiny page pools (constant preemption), small step budgets
  (chunking) and slot/page reuse. 5 configurations x 3 seeds x 400 requests: all identical, no leaks.
- Scheduling (vLLM V1-like): every running sequence with one token left to feed decodes; prefills
  fill the remaining token budget in admission order, one chunk per sequence per step; pages are
  allocated as positions advance; when the pool cannot cover a step, the most recently admitted
  sequence is preempted (cache dropped, recomputed later from prompt + output). Admission wants pages
  for the first chunk plus one page of headroom per running sequence.
- Sampling on the host from the step's fp32 logits (penalties, temperature, top-k/top-p/min-p, per
  request seeded RNG, raw logprobs). Logits come back through a pinned buffer sized for the logit rows
  a step can request (not max_tokens rows: 0.53 GB at 1024).
- HTTP: cpp-httplib (header-only, via pkg-config so its CMake config does not pull in OpenSSL), one
  thread per in-flight request; deliveries go engine thread -> per-request channel -> HTTP thread,
  which runs detokenizer -> stop strings -> DSV4 parser. A disconnected client or a matched stop
  string aborts the request in the engine.
- First served run (4 slots x 32k, 1024-token steps, EXL3 3.04 bpw experts, load 21 s): correct
  answers with reasoning, parallel tool calls in one streamed response with finish_reason
  tool_calls. Decode: 16.4 tok/s for one stream; 4 streams 29.7 tok/s aggregate (7.8 each) — the step
  is launch/latency-bound (a batch-1 token reads ~2.5 GB of weights, ~4 ms at the cards' bandwidth,
  but takes 58 ms), and per-chunk attention/compressor work plus more distinct experts (and more APU
  round trips) grow with the batch. Next: profile the batch-4 decode step.

## 2026-10-11 — long context, part 1: fp4 indexer keys, int8-WMMA scores, bounded score scratch

Baseline before (bf16 keys, fp32 scalar scores over a [T, max_seq/4] fp32 buffer): the indexer cost
grew linearly with depth (0.58 -> 1.6 s per 4096-token chunk from 4k to 8k), the score scratch was
N x keys x 4 bytes (2 GB at 512k).
- Keys are stored as the reference already quantizes them: Hadamard, then e2m1 codes + one e8m0
  scale per 32 (68 B/key instead of 256). `store_rows_fp4` round-trips to exactly fake_quant_fp4's
  bf16 values (0 differences on 20M values). The reference indexer reads the same pool.
- Fast scores: queries -> int8 codes (2 x e2m1) + scales; int8 WMMA per 32-value block (exact integer
  dot), scaled by the two exact power-of-two scales; then the reference's bf16(dot), ReLU, bf16(x w_h)
  and in-order fp32 head sum. Scores are bf16 [T, keys]; top-k is a 16-bit radix select. Against the
  reference kernels (`unholy-indexer-test`, up to 131k keys, shuffled block tables): scores and top-512
  indices identical. Queries go through a 256 MB score buffer in sub-batches.
- Measured (1 slot, 4096-token chunks): 16k 679 tok/s, 64k 585 tok/s, 128k 521 tok/s average; the
  last 128k chunk 9.3 s (442 tok/s). Decode at 128k depth 65 ms/token. Profile of that chunk: indexer
  2.1 s (~22 TOPS: the per-element scaling/epilogue dominates), EXL3 GEMM 2.0 s on the cards and 3.2 s
  on the APU (now on the critical path: 1.5 s of card waits), remote-row copies 0.95 s, sparse
  attention 0.73 s (HCA: one block per query re-reads every compressed row).

## 2026-10-11 — routing profile from our own traffic; EXL3 grouped GEMM v2

- The placement profile inherited from dsv4-engine (`data/expert_profile_dsv4engine_code32k.txt`)
  matches this model's routing only in the early layers: per-layer correlation with our own histogram
  of the same code prompt is 0.88-1.0 for layers 0-12 and 0.13-0.49 from layer 16 on. With it, a
  4096-token prefill chunk sent 11-50% of a layer's pairs to the APU (planner prediction: 7%) — the
  copies and the APU's GEMMs were on the critical path. Our histogram's top-128 experts per layer hold
  86% of the routing mass vs 75% for the old profile's top-128.
- The device MoE count now also accumulates a running per-layer histogram (one more atomic per pair,
  no extra launch); `unholy serve` exposes it at GET /unholy/routing_profile (profile format) with
  POST /unholy/routing_profile/reset, and `tools/ref/routing_workload.py` drives a mixed workload
  (code, math, English/Chinese/other-language writing, tool calls, long documents, thinking on/off)
  through a server to produce the profile the planner should use.
- EXL3 grouped GEMM v2 (prefill): 128-row tiles (an expert's weights are decoded once per step for
  up to 128 rows instead of once per 64), each wave decodes exactly the 16-column trellis slice it
  multiplies into wave-private LDS (no block barrier for B), a lane decodes its 8 consecutive trellis
  steps from three words with funnel shifts and the mul1 byte sum is one `v_dot4_u32_u8`, next-slab
  loads are in flight during the current slab. Same k-step order and epilogue as v1: outputs
  bit-identical on gfx1201 and gfx1151 (`unholy-exl3-gemm-test`); 3.1-4x faster (~300 G weights/s,
  44 TFLOPS on a shared card). The decode-path GEMV uses the same lane decode (agrees with the GEMM to
  rel L2 1e-4; it sums in a different order).

## 2026-10-11 — two steps in flight (card pipelining)

`StepModel` is asynchronous (submit -> ticket, wait in order, up to max_inflight). The resident model
keeps two steps in flight: card 0 runs step k + 1 while card 1 finishes step k. Each in-flight step
has its own pinned staging (ids/positions/logit rows, block tables, the card 0 -> 1 residual, logits)
and three events (residual written, residual consumed, logits ready); block tables are uploaded on the
card streams with the next step instead of synchronously; each card has its own channel to the APU
experts (stream, flags, pinned staging, scratch; weights shared), so both cards can have expert work
on the APU at once. Stream order keeps every step after the ones submitted before it, so consecutive
prefill chunks of one sequence overlap safely. The engine plans only sequences that are not waiting
for logits, advances positions at submission, preempts only sequences with nothing in flight, and
defers aborts of in-flight sequences to their step's completion (fake-model scheduler test at depth 1
and 2: all outputs identical, no leaks). UNHOLY_INFLIGHT=1 restores one step at a time.

Measured (16k context, 4096-token chunks, mixed-traffic profile, EXL3 GEMM v2): prefill 1087 tok/s
one step at a time, 1680 tok/s with two in flight (679 tok/s earlier today with v1 and the old
profile); single-stream decode 53.4 ms/token (18.7 tok/s).

Full-model KL vs the oracle moves with placement (old profile 2.445e-2, mixed profile 2.833e-2) but
per-layer errors are the same on a card and on the APU (layer 5: 3.255e-2 vs 3.262e-2, layer 16
3.727e-2 vs 3.701e-2, layer 30 1.842e-2 both) and the grouped GEMM agrees across gfx1201/gfx1151 to
bf16 rounding: the spread is accumulation-order noise amplified through 43 layers of fp8 activation
quantization. Single-prompt KL numbers need averaging (several prompts/perturbations) before they
can rank quantizations.

`unholy plan` prints the placement for a geometry without loading. With the mixed profile and bf16
KV: native MXFP4 experts at 4 x 512k fit (APU 109 GB of 129.5 GB free; 52.8% of routing on the cards;
1 x 512k: 59.7%); EXL3 3.04 bpw at 4 x 512k: 63.9%. fp8 KV storage would free ~6 GB more on the cards.

## 2026-10-11 — native MXFP4 experts on grouped kernels; depth curves to 512k

Native experts get the EXL3 path's device dispatch, 128-row tiles and APU channels with their own
kernels (`src/kernels/mx_moe.hip`): activations fake-quantized to fp8 per 128 into bf16 (exact), each
wave dequantizes the weight slice it multiplies into LDS through a 256-entry byte -> bf16-pair table
with the power-of-two scale added to the exponents (per half: a negative offset borrows), bf16 WMMA with
fp32 accumulation. The GEMV takes the unscaled values and scales each 32-block's partial sum (exact for
powers of two). Weights are repacked at load into a tile-major layout (one contiguous 512-byte read per
wave and 64-input slab; lossless; the reference GEMM reads it too and gives identical bits). Against
the reference A8W4 GEMM: rel L2 ~1e-6, i.e. ulp-level, on gfx1201 and gfx1151. Card: GEMM 38 TFLOPS,
GEMV 383 GB/s; APU: GEMV 212 GB/s (near its bandwidth). Native load (167.8 GB, with repack) 41.7 s.

`unholy-curve` (one sequence, 4096-token chunks, two in flight, decode probes; logs in docs/bench/):

| depth | prefill EXL3 | prefill native | decode EXL3 | decode native |
|---|---|---|---|---|
| 4k | | | 52.5 ms | 50.3 ms |
| 16k | 2001 tok/s | 1693 tok/s | | |
| 32k | 1763 | 1574 | 55.9 ms | 53.6 ms |
| 128k | 1299 | ~1150 | 62.0 ms | 59.8 ms |
| 256k | 1090 | 1037 | 71.4 ms | 68.0 ms |
| 512k | 850 | 819 | 84.9 ms | 82.5 ms |
| average 0-512k | 1044 | 971 | | |

Native (lossless, 59.7% of routing on the cards at 1 x 512k) decodes slightly faster than EXL3 3.04
bpw (71.9%) — MXFP4 dequantization is far cheaper than trellis decoding — and prefills 3-15% slower
(more APU work). Both hold the >= 500 tok/s prefill target at 512k depth.

## 2026-10-12 — GPU memory fault serving native experts: investigation

Symptom: with native MXFP4 experts the server faults (HSA_STATUS_ERROR_MEMORY_FAULT, amdgpu "[gfxhub]
page fault ... PERMISSION_FAULTS 0x3, client TCP") after 2-25 sequential chat requests; every later
request then fails. 9 of 9 runs. With EXL3 experts: 0 faults (3 runs of 48 requests).
Always physical card c7 (logical card 1), always the first kernel of card 1's part of a new prefill
(hc_pre_tiled), always an address that is a card-1 weight with bit 44 cleared — attributed exactly:
inside layer 25's hc_attn_fn at +0xb000, the tensor that kernel reads first (UNHOLY_DEBUG_TENSORS).
Ruled out, each by a run:
- our host pointers (UNHOLY_CHECK_PTRS: unchanged), out-of-bounds indexing (the access is in-bounds
  on a live allocation), kernel scratch (removed from all kernels: hc_pre_tiled's runtime-indexed
  Sinkhorn arrays, compress_rows' 2 x 128 locals, a staging array in the bf16 WMMA GEMM — numerics
  unchanged, KL 2.445e-2), pipelining (UNHOLY_INFLIGHT=1 faults), kernarg placement
  (HIP_FORCE_DEV_KERNARG=0 faults), VRAM headroom (3 GB margin faults; `--card-margin-gb`, default
  now 3), memory pressure / reclaim (vmstat reclaim and swap counters and PSI unchanged during a
  faulting run; EXL3 under 46 GB of stress-ng pressure: clean), idle power states (card kept busy by
  a second process: faults), async copies (all per-step staging now by kernels pulling from /
  pushing to pinned memory, the per-layer APU metadata by one kernel: faults), VRAM cells (pattern
  test, 6 passes x 33.6 GB on both cards: 0 bit errors), PCIe (no AER errors).
- Any kernel or copy serialization (AMD_SERIALIZE_KERNEL / AMD_SERIALIZE_COPY) hides it.
- Swapping the cards' roles (HIP_VISIBLE_DEVICES=1,0,2: c3 runs layers 22-42 + head, c7 layers 0-21):
  0 faults in 48 requests (concurrency sweep on that order running for confirmation).
Working hypothesis: a timing-dependent hardware/firmware-level problem on c7's path in the card-1
role (one bit of a kernel-argument pointer, bit 44, arrives cleared). Kept from the investigation:
no kernel uses scratch, staging by kernels, `stage_remote`, `--card-margin-gb`, the debug dumps.

## 2026-10-12 — the fault: a register bit flip on c7 under the 230 W power cap

Follow-up to the entry above. Two more facts settled it:
- One hardware queue per card (GPU_MAX_HW_QUEUES=1, as the vLLM service runs) still faulted (21
  requests in). The kernel log also shows vLLM itself faulting on these cards before unholy existed
  (c7 on 2026-09-28, c3 on 2026-10-04) — but those were ordinary out-of-bounds reads (runs of
  consecutive pages; c3's other faults were null pointers from an early engine build), not this.
- A canary in hc_pre_tiled (UNHOLY_CANARY, kept): the kernel gets the weight pointer twice, once more as
  an integer at the end of the argument list, and checks every load address against one recomputed
  from the copy. At kernel entry the two always agreed. The faulting address is fn + 0xb000: 44 of the
  64 K-chunks had been read with the same base before it. In the ISA (gfx1201) the base lives in a VGPR
  pair that the loop only increments by 0x400 (v_add_co_u32 / v_add_co_ci_u32, carry via s0); nothing
  in the code can clear bit 12 of the high dword. A register changed under the kernel, on c7 only
  (13/13 faulting runs; never on c3, including when c3 ran the identical role).

c7 also runtime-suspends/resumes (BACO, "PSP is resuming") about once a minute when idle — 340 times
the day before — which c3 never does, and the unclean host reset at 05:08 on 2026-10-12 followed a
burst of those cycles every ~6 s.

Resolution: the LACT power cap (230 W on both cards; stock 300 W) was removed in /etc/nixos. At 300 W:
96/96 requests with the plain kernel and 48/48 with the canary, no fault and no canary event (at 230 W
every run faulted within 2-25 requests). Working explanation: power-limit throttling transients push
the weaker chip (c7) past its timing margin. If it recurs, the canary names the wave's HW_ID1 (SE / SA
/ WGP / SIMD) so the unit can be masked with HSA_CU_MASK.

## 2026-10-12 — prefix caching, vLLM-style KV settings, 1M context

KV settings follow vLLM: `--kv-tokens` is one pool of 256-token pages shared by all requests,
`--max-model-len` caps a request, `--max-num-seqs` caps concurrency (a slot is ~17 MB of fixed state:
the attention window and compressor rings). ~5.8 KB per token of pool (bf16 compressed rows + fp4
indexer keys), split across the cards; every GB displaces ~80 hot experts.

Prefix caching (engine.cpp): pages are named by a hash chained over the tokens to their end; a finished
request's pages stay cached until evicted (LRU, a sequence's tail before its head). DSV4's unpaged
per-slot state means a prefix is reusable only where that state was saved: snapshots (window rows +
ratio-4 compressor and indexer rings, ~7 MB, pinned host memory, `--prefix-snapshots`) at the prompt's
last page boundary, every 4k/16k/64k of prefill, and every page decoding completes (each superseding the
previous). The ratio-128 rings are skipped: at a page boundary (a multiple of 128) no later block reads
them. Restores and saves are StepInput ops run in stream order on each card around the step's layers.
Verified: the scheduler test's fake model checks restored state against the slot's pages; on DSV4 a
16.8k-token prompt repeated, with a different question, and as the next chat turn reused 16,640 tokens
each, with logprobs identical to uncached runs.

1M context: the rope tables (512 B per position per card) were not in the plan — 0.5 GB per card at 1M,
which overflowed the 1.5 GB default margin; now planned, and the planner's default margin is serve's
3 GB. Two 1M requests (a 2M-token pool) fit native: 6 GB of cache per card, 50% of routing on the cards.

## 2026-10-12 — DSpark speculative decoding

The checkpoint's mtp.* is DeepSeek's DSpark block drafter (model.py DSparkBlock): 3 stages, each a full
Block (HC, attention, 256-expert MoE) with compress ratio 0. Its attention keys are a 128-position
window of "main-KV" — each stage's wkv/kv_norm/rope/fp8-sim of main_x = main_norm(main_proj(the
target's HC means after layers 40-42, concatenated)) — in slot order, plus the whole draft block
(bidirectional). A draft block is [the last sampled token, 4 noise tokens] at the next 5 positions; the
last stage adds hc_head, norm, the target's LM head, a rank-256 Markov bias of the previous token and
an argmax chain (temperature 0), and a confidence head.

Implementation:
- Weights: dense parts on card 1 (0.74 GB), routed experts on the APU (10.3 GB, own channel).
- Every target step records the HC means of the target layers for each chunk's last <= 128 rows and
  writes the stages' main-KV of them (so prefill fills the windows, verify steps extend them).
- Verify chunks: a decoding sequence feeds its token plus k drafts with logits for every row. Before
  the chunk, every per-slot ring the positions overwrite (attention windows, ratio-4 and ratio-128
  compressor rings, indexer rings, the stages' windows) is backed up on both cards; after the drafts
  are judged, the rejected positions are restored (paged rows need nothing: they are rewritten before
  any later query can read them). Snapshots for prefix caching include the stages' windows and are kept
  only when the whole chunk was accepted.
- Acceptance: speculative sampling with a deterministic proposal — accept draft d with probability
  p(d) under the request's own sampling (penalties, temperature, top-k/p, min-p), else draw from p
  without d — so outputs are distributed exactly as without speculation (greedy: accept iff argmax).
- Drafts run on a card-1 stream of their own (scratch, APU channel) after the verify step completes,
  alongside the other in-flight step.

Verified: tools/ref/oracle/dspark.py (the reference stages on the oracle's hidden states) vs
unholy-dspark-test on the same inputs — identical drafts at 7/7 positions, confidences within 0.03,
logits rel L2 1-4%; the scheduler test with a fake drafter (25% wrong tokens) gives outputs identical
to non-speculative greedy in every configuration (with prefix caching, preemption, aborts); on DSV4
greedy outputs with and without speculation are identical. 4 drafts, single stream, short context:
code 19.9 -> 35.1 tok/s, prose 18.9 -> 23.0 (52% of drafts accepted, 3.1 tokens per verify step).

## 2026-10-12 — vision

The checkpoint is DeepSeek-V4-Flash-Vision-Exp: a 32-layer ViT (1024 wide, 16 heads, 2-D rope,
bidirectional), an Aligner (3 x 3 unfold, linear, GELU, linear) and image tokens (vocab + type) laid out
by build_image_block in an "N-layout" that puts each 2 x 2 patch square into one 4-token compression
block. Image tokens embed as the aligner rows (or learned start / pad / newline / end rows), route with
gate.bias_vl (by score on the hash layers too), and see their whole image span both ways in the
sliding-window attention (512 window entries); compressed attention stays causal.

- Preprocessing on the CPU, bit-exact with the reference: libjpeg-turbo with PIL's decoder settings,
  libpng (16-bit grey clipped to 255 as PIL's convert does), libwebp, Pillow's bicubic resampler ported
  (fixed point, two passes), load_image's sizing / pad / normalisation. 60 synthetic + 2 example images:
  identical bf16 patches and blocks; 12 multi-image prompts expand to identical ids; 4134 random
  conversations with image blocks encode identically (text_test / chat_compare.py, vision_compare.py).
- The ViT runs on the APU (1.2 GB with scratch): bf16 WMMA GEMMs with the reference's rounding points,
  a flash-style attention. Against the reference block rows: rel L2 2.2e-2 mean — the same as the
  reference's own CPU vs GPU difference (2.24e-2, worst rows 0.2-0.3; the ViT is that sensitive).
  Encode: 870 ms on the APU, 409 ms on an R9700, for a 42 x 61 patch grid.
- End to end (oracle run.py --vl-request, teacher-forced on our output): identical 329- and 119-token
  prompts; our greedy tokens are the reference's argmax at 3/3 and 40/41 positions, median |logprob
  difference| 0.010; the one difference (the first word) also flips between two reference runs whose
  only difference is where the ViT ran.
- The engine keeps an image block whole in one chunk, keys page hashes on the image's hash (identical
  type ids otherwise), never resumes a cached prefix inside a block (its tokens see the whole block), and
  refuses blocks that cannot fit a step.

Memory: three 512k requests + DSpark + the vision tower did not fit (a pinned allocation failed at load).
The card margin goes 3 -> 2 GB (the faults it was raised for were the power cap) and son-of-anton keeps
64 prefix snapshots: loads with ~2 GB more experts on the cards (53.3% of routing vs 51.6%); GSM8K at
concurrency 4 (warm cache): 44.5 tok/s aggregate, 97.9%.

## 2026-10-12 — PLX switch; compact card <-> APU expert transport

Topology since today: both R9700s behind a PEX88048 switch card (2 x16 downstream ports, Gen4) in the M.2
slot at root port 00:02.5, i.e. one Gen4 x4 uplink to the Strix Halo shared by both cards. `unholy-probe`:

| path | before | behind the switch |
|---|---|---|
| card <- card, kernel pull (verified) | 7.1 GB/s (root complex) | 26.3 / 26.7 GB/s |
| host -> card / card -> host, one card | 7.16 / 6.96 GB/s | 4.6-5.2 / 5.5-5.8 GB/s |
| both cards at once, h2d / d2h aggregate | 14.4 / 14.0 GB/s | 6.1 / 5.2 GB/s |
| card <- APU memory pull / card -> APU push | 7.2 / 7.2 GB/s | 5.8-6.0 / 5.5-5.6 GB/s |

Kernel parameters (hosts/son-of-anton): `pci=disable_acs_redir=pci:1000:c012` — with ACS redirect on (the
kernel requests it whenever an IOMMU driver is loaded, `iommu=pt` included) card -> card reads are sent
to the root complex, which does not route them back down the port they came from: the reading queue
wedges (killing it forced a GPU reset that took the box down). Bring-up also hit "device lost from bus"
dropouts and data-fabric sync-flood resets under load (`x86/amd: Previous system reset reason`); they
were a physical short on the new boards, not software. 25 min of sweep + 512k curve + a 3-minute
card <-> card soak at 26 GB/s afterwards: nothing lost.

Cost, old transport (`unholy-curve` 1 x 512k, docs/bench/curve-native-1x512k-plx-old-transport-*.txt):
average prefill 971 -> 851 tok/s. A rocprofv3 trace of a 64k prefill showed why: per card, 28% of the
time in `copy_remote_rows` (bf16 pair rows to the APU and back over the shared uplink), 28% waiting for
the APU, ~30% computing; the APU computing 58% of the time.

Transport changes (native MXFP4 experts; `moe_device_dispatch`, `ApuExperts`):
- Inputs: one fp8 row per token with remote pairs (e4m3 codes + an e8m0 exponent per 128 —
  `quant_remote_tokens`) instead of one bf16 row per remote pair: the exact values `mx_xform_in` computes
  on the APU, which rebuilds them (`mx_xform_in_q`). ~5x fewer bytes out; bit-identical (code1024
  oracle: 133,416,960 / 133,416,960 logits). The first version launched a block per 128-value segment
  with a system-scope fence each (131k fences per layer) and was slower than the copy it replaced
  (11.0 vs 8.7 s per card); grid-stride with one fence per block fixed it.
- Outputs: the APU sums each token's remote pairs (fp32, ascending expert order, `remote_combine`) and
  returns one bf16 row per token (~2.6x fewer bytes back); the card adds it after its local pairs
  (`moe_combine_partial`). This rounds the remote share to bf16 once and makes the sum order depend on
  placement (the reference model accumulates every expert's output in bf16). code1024: KL vs the oracle
  1.027e-2 -> 1.000e-2, top-1 96.32% -> 95.64% (7 near-ties of ~1032); GSM8K, 200 questions,
  concurrency 8: strict 0.615 vs 0.645 (CI +-0.067), flexible 0.980 vs 0.980.
- Switches: `UNHOLY_REMOTE_COMBINE=0` (pair rows back, bit-exact), `UNHOLY_REMOTE_Q=0` (the old
  transport); `unholy-resident-test --out` saves logits for a later `--ref`.

| 128k curve, same plan (60.5% on cards) | avg prefill | decode at 4k |
|---|---|---|
| old transport | 1018 tok/s | 57.1 ms |
| fp8 inputs (bit-identical) | 1114 tok/s | 57.2 ms |
| fp8 inputs + combined outputs | 1254 tok/s | 56.4 ms |

512k curve (docs/bench/curve-native-1x512k-plx-2026-10-12.txt): average 1062 tok/s (971 before the
switch), 896 tok/s at 491k. GSM8K production sweep (3 x 512k, DSpark 5, vision): concurrency 1/2/4/8
22.2 / 36.5 / 45.5 / 56.3 tok/s aggregate (before the switch 20.1 / 34.2 / 39.9-44.5 / 50.8).

Open: raw single-sequence decode is ~5.7 ms/token slower behind the switch with the same code (58422ca:
51.6 -> 57.3 ms at 4k), ~130 us per layer. The card links stay at Gen4 throughout decode (not PCIe DPM);
the suspect is per-round-trip latency of the per-layer card <-> APU handoffs (flags, polls, copies)
through the switch. With DSpark (several tokens per round trip) the sweep shows no decode loss.

## 2026-10-12 — prefill order: shortest remaining first

In production a 164k-token prompt (1291 tok/s, TTFT 127 s) held up a continuation that arrived 1.6 s
later with 18,176 of its 18,678 tokens cached: ~500 tokens of prefill, TTFT 115 s. Prefill chunks were
packed in arrival order, so the long prompt took the whole 4096-token budget every step. vLLM's V1
scheduler does the same by default (running requests in admission order, each taking min(remaining,
budget); checked at vllm 189bbfb), and the old vLLM service here set nothing else; its only remedy is
`--long-prefill-token-threshold` (off by default; `-adaptive` floors it at budget / requests).

Now decodes still go first, then prefill chunks shortest-remaining-first (`--prefill-order shortest`, the
default); the earliest-submitted prefill keeps a quarter of each step so a stream of short prompts cannot
stall it. `--prefill-order fcfs` is the old behaviour, and `--long-prefill-token-threshold[-adaptive]`
are vLLM's knobs with vLLM's semantics. A prefill chunk that gets less than its whole remainder ends on a
page boundary (snapshots are only taken there). unholy-sched-test runs the whole matrix in all three
modes and races a 2000-token prompt against a 20-token one a step later (64-token steps): shortest
31 / 1 (finish steps, long / short), fcfs 31 / 31, fcfs + threshold 16 32 / 2.

## 2026-10-14 — decode profile; the shared expert overlaps the APU

rocprofv3 kernel traces (`unholy-bench`, 4k context, native experts, mixed profile; the profiler adds
8-15%) of where a decode step goes. Batch 1, 60 ms profiled (52.6 unprofiled): each card ~30 ms, of
which ~22 ms kernels: dense GEMVs 20 ms per token (wq_b / wo_a / wo_b, 32 MB each, at 370-400 GB/s of
the 595 measured), single-workgroup kernels 15.4 ms (sparse_attn 87 us avg, 145 us on CSA layers;
moe_plan 58 us x 43; hc_post 28 us x 86; hc_finish 16 us x 86; rmsnorm 7 us x 181; moe_combine_partial
23 us x 43; topk 38 us x 21), ~1350 launches per card per token with a median 4.6 us gap between
kernels (~13 ms per token). Host enqueue is ~2 ms per card (UNHOLY_STEP_TIMING), so the step is
GPU-side latency, not launch submission. Batch 1/2/4/8 steps: 52.6 / 68.7 / 101.7 / 168.4 ms; at batch
8 the per-chunk attention alone is 344 single-workgroup launches (28.8 ms) and top-k 168 (6 ms). The
APU's MXFP4 GEMV at decode runs 16-32 workgroups (~100-135 GB/s). Single-stream DSpark (serve, greedy):
a cycle is ~100 ms profiled: verify 87 (43 + 44 on the cards), draft 10.5 (3 of it waiting for the
APU), host ~2. Prefill at 48-64k: the cards are busy 48-57% and wait for the APU 6.3 / 7.1 s of a 16.8 s
window; the APU is 57% busy (its MXFP4 GEMM ~11 TFLOPS).

`unholy-resident-test --batch N` now checks batching bit-exactly: N sequences decoded alone, then
together, then two as 4-token verify chunks. Baseline: identical in both; with 4 x 4-token chunks
(16 rows, past kGemvMaxRows) the GEMM kernels sum in another order than the GEMVs and nothing matches,
so verify steps of several streams are not bit-identical to single-stream ones (pre-existing).

First change: the shared expert was computed after the card stream joined the APU's experts; it is
now queued between the APU request and the join (moe_device_dispatch / moe_device_join), so it runs
while the APU works. Same kernels on the same inputs: logits bit-identical (code1024, both
geometries, batch checks). Decode 52.6 / 68.7 / 101.7 / 168.4 -> 49.3 / 65.2 / 96.4 / 157.8 ms at
batch 1 / 2 / 4 / 8; 4k prefill 1187 -> 1240 tok/s (one chunk), 1827 -> 1891 (8 sequences).

## 2026-10-14 — single-workgroup decode kernels spread out

At decode a token's elementwise kernels ran one workgroup per row: `hc_post` 28 us, `hc_finish` 16 us
(a 20-iteration Sinkhorn on one thread, then the mix), `moe_combine_partial` 23 us, `rmsnorm` 7 us, and
`moe_plan` 58 us (thread 0 walking 256 experts twice). Now:
- `rmsnorm_slices`, `hc_post_slices` (`src/kernels/elem_fast.hip`), `moe_combine_partial` and the mix of
  `hc_finish`: (row, slice of dim) workgroups, ~64 per launch at small T; the per-element arithmetic is
  unchanged (rmsnorm's sum of squares is repeated per slice with the same partition; out of place only).
- `hc_finish`'s Sinkhorn on 16 lanes of wave 0, one comb entry each, row / column sums through shuffles in
  index order: the same operations per value as the serial loop.
- `moe_plan`: block-wide scans (pair rows and tiles of the resident and the remote group) instead of the
  serial walk: same offsets and tile order.
`UNHOLY_REF_ELEM=1` keeps the reference rmsnorm / hc_post. Logits bit-identical (both geometries, batch
checks). Decode 49.3 / 65.2 / 96.4 / 157.8 -> 42.6 / 58.1 / 89.6 / 151.0 ms at batch 1 / 2 / 4 / 8.

## 2026-10-14 — one attention launch per layer for all chunks

A step's chunks ran attention (one workgroup per query) and the indexer's top-k (one workgroup per row)
chunk by chunk: at batch 8, 344 serialized single-workgroup attention launches (28.8 ms of the step) and
168 top-k launches (6 ms). Now each layer runs the indexer for every chunk first (score rows of several
chunks share the score buffer; `topk_bf16_batch` takes up to 32 row groups with their own n and k, chunk
c's top-k at idx + t0 * idx_topk), then one `sparse_attn_wmma_batch` launch over all chunks (a table of
up to 32 chunks in the kernel arguments; each block finds its chunk and runs exactly the single-call
code), then the window rings. Chunks of one step are of different slots, so no chunk reads another's
ring update (a step with two chunks of one slot keeps the old order; `UNHOLY_ATTN_PER_CHUNK=1` forces it).
Logits bit-identical (both geometries, batch checks), indexer kernel test unchanged. Decode at batch
1 / 2 / 4 / 8: 42.6 / 58.1 / 89.6 / 151.0 -> 42.5 / 53.6 / 76.0 / 119.5 ms.

## 2026-10-14 — the W8A8 GEMV was ALU-bound on the fp8 decode

`unholy-gemv-test` (new) times the decode GEMVs on the model's shapes with the weights streamed from DRAM
(enough copies to defeat the 64 MB last-level cache; one 32 MB matrix repeated runs from cache at
>1.3 TB/s) and checks that outputs do not depend on the columns per wave. The bf16 GEMVs ran at 500-630
GB/s, the W8A8 one at ~380 GB/s with one row and ~85 GB/s with eight: every weight and (per row) every
activation went through the bit-level e4m3 decode (~7 ALU ops each). `dev::e4m3x4_to_f32` uses gfx12's
`v_cvt_pk_f32_fp8` instead (exact: all 254 finite codes give the bit decode's values, checked by the
test; gfx1151 keeps the bit decode). wq_b / wo_b / wo_a (32 MB each) now stream at 540-556 GB/s with one
row and 370-420 GB/s with eight (wo_b at M = 8: 399 -> 79 us). More columns per wave (sharing the
activation decode) only cost occupancy once the decode was cheap: one column per wave stays the default.
Logits bit-identical. Decode at batch 1 / 2 / 4 / 8: 42.5 / 53.6 / 76.0 / 119.5 -> 37.8 / 45.3 / 59.0 /
89.0 ms.

## 2026-10-14 — verify steps of several streams: GEMVs up to 16 rows, MoE GEMVs up to 64 tokens

Production decodes with DSpark verify chunks of up to 6 rows per stream, so two or more streams are past
8 rows, where every projection switched to the WMMA GEMMs (64 / 128-row tiles). `unholy-bench --verify 6`
at batch 1 / 2 / 4 / 8: 66 / 172 / 243 / 417 ms per step. At 12 rows the MoE ran `mx_grouped_gemm` on
~45 experts of 1-2 rows each (cards 52 ms, APU 91 ms per step, 705 us per call on the APU), and the dense
WMMA tiles are latency-bound at small M (few workgroups: wq_a / wkv / the router 76-87 us whatever M;
unholy-gemv-test now times both).
- MoE: steps of up to `mx_gemv_max_tokens()` tokens (64, `UNHOLY_MX_GEMV_TOKENS`) plan their experts in
  8-row tiles (`moe_plan` tile_rows) and run `mx_grouped_gemv` on the cards and the APU (bandwidth-bound:
  ~570 GB/s on a card, ~190 on the APU at 12 rows). A row's GEMV result does not depend on its tile.
- Dense: the GEMVs take up to 16 rows (`kGemvMaxRows`; templated on a row capacity of 8 / 16 / 32 so small
  M keeps its registers). At 12-16 rows they beat the WMMA tiles on every dense shape but wq_b and the
  head (net ~240-430 us per layer); past ~24 rows the tiles win.
Rows of steps up to 8 tokens are unchanged (logits bit-identical, both geometries). Rows of 9-16-row steps
now get exactly the single-stream GEMV values instead of the GEMM ones: `--batch 4`'s 4 x 4-token verify
chunks (16 rows), which matched nothing before, are bit-identical to the solo runs. Verify-6 steps at batch
2: 171.7 -> 115.3 ms; batch 4 / 8 (24 / 48 rows): 243 / 417 -> 182 / 317 ms (MoE only).

## 2026-10-14 — two decode micro-batches

With every decoding sequence in every step, a pure-decode workload had one step in flight: the next step
found all of them waiting for logits, so card 0 idled while card 1 finished and the reverse. Now
(`--decode-split`, default on) a step takes half of the decoders, counting the ones in the step already
in flight (ceil((in flight + eligible) / 2)); the rest are not waiting for logits, so the next step takes
them. Two halves then alternate, each submitted when the other's predecessor completes. The count matters:
"half when nothing is in flight" alone left a 1 + 7 split when one request started before the others
(the server trace showed T = 4 steps once balanced). unholy-sched-test: all ok.

Server, production flags, N concurrent greedy requests of 256 tokens (ignore_eos), aggregate tok/s at
N = 1 / 2 / 4 / 8 (this build, before -> after the split):
- no speculation: 27.2 / 49.0 / 64.5 / 80.8 -> 27.2 / 48.9 / 69.6 / 90.0
- DSpark 5 drafts: 42.1 / 51.5 / 62.2 / 75.2 -> 41.6 / 50.6 / 66.8 / 84.7
Under 8 streams the cards now run kernels ~40% of the time and wait for the APU ~41%; the APU is 65% busy
and its expert GEMVs share its ~200 GB/s between the two cards' channels (385 us per call vs ~200 alone,
unholy-mx-test --dev 2): multi-stream decode is bound by the cold experts' bytes on the APU.

## 2026-10-14 — attention: a bank conflict in the V transpose; DSpark stages on the WMMA kernel

`unholy-attn-test` (new): the WMMA attention against the reference kernel on decode-sized calls over a
paged pool (window, CSA top-512, HCA all rows, at 4k / 128k / 512k), time per call. A query's keys go
through the workgroup 16 at a time, ~3 us per block whatever the pointer loads (prefetching the next
block's rows during the current one, kept, changed nothing measurable): the staging wrote V transposed
(`sVt[dim][key]`, 2-byte stores) from lanes holding one key and dims 32 apart, so the 16 lanes of a
half-wave hit one LDS bank on each of their 32 stores. Lanes now hold the 16 keys of one dim range (same
LDS contents): CSA 134 -> 81 us, window 40 -> 26, HCA at 128k 205 -> 117, at 512k 812 -> 434 us per call.
Packing two keys per 4-byte store through a lane shuffle was slower (95 us CSA). Four workgroups per query
(one per 16-head group, each reading every key) was no faster: the block loop is latency-bound, not
compute-bound.

DSpark stages used the reference attention (one block per head per query: 350 us per stage); the WMMA
kernel now has the stage key order (ring slots in order, then the draft block): unholy-dspark-test 7/7
drafts identical to the reference, logits rel L2 1-4% as before.

Logits bit-identical (both geometries, batch checks). Decode at batch 1 / 8: 38.1 / 89.2 -> 36.4 / 87.3 ms;
4k prefill 1303 -> 1371 tok/s (one sequence), 1948 -> 2046 (eight).

## 2026-10-14 — top-k for a few long rows on several workgroups

At decode the indexer's top-512 ran one workgroup per row: four serial passes over the row with 256
threads, then the threshold ties collected by one wave walking the row in order. `unholy-indexer-test`
(now timing it, as the model calls it): 224 us at 32k keys (128k context), 727 at 131k (512k), 1538 at
262k (1M) per row: at 512k, ~15 ms of each token across the 21 CSA layers. `topk_bf16_batch` now takes
scratch and, for up to 64 rows of which the longest has >= 8192 keys, runs five launches of (row,
2048-key slice) workgroups: per-slice high-byte histograms; their sum -> the threshold's high byte and
per-slice low-byte histograms of that bin; the threshold and per-slice counts above / at it; the
candidates, ties ranked by index through the slices' prefix; a sort per row. Same selection, ties and
order as the one-workgroup kernel (indexer test: 0 differing indices at 8k-262k keys, T = 1 / 2 / 3 / 6 /
37). 32k / 131k / 262k keys: 224 / 727 / 1538 -> 57 / 72 / 94 us. The scratch is the score buffer past the
pending rows (a buffer of its own moved experts in the plan: with the APU's bf16 combine, and even
without it, a different placement changes the logits).

## 2026-10-15 — fewer launches per layer; long-context decode at 20 tok/s

A token ran ~63 kernels per layer on a card, ~4.5 us apart. Removed, bit-identically:
- `gemv_w8a8_parts` / `gemv_bf16_f32out_parts`: up to three GEMVs on one input in one launch (each wave
  finds its part, then the single GEMV's code): wq_a + wkv, wq_b + the indexer's wq_b (both on the
  quantized qr), the shared expert's w1 + w3, each compressor's wkv + wgate. `UNHOLY_NO_GEMV_PARTS=1`
  launches them separately: same bench, 37.5 / 60.7 / 92.5 -> 36.7 / 59.2 / 91.0 ms at batch 1 / 4 / 8.
- the bf16 -> fp32 copies of the compressor and router inputs, which only the reference GEMMs read;
- the per-layer memset of the MoE counts: moe_plan leaves them zero for the next layer.

`unholy-curve` to 512k with tonight's kernels (the attention staging, the long-row top-k): decode 36.6 /
38.8 / 40.8 / 44.6 / 48.9 ms per token at 4k / 32k / 128k / 256k / 512k (before tonight 56.7 / 58.4 /
64.2 / - / 87.6): 20.4 tok/s at 512k without speculation. Prefill average 0-512k 1245 tok/s (1062).
- `rmsnorm_quant_slices`: the attention, q-LoRA and FFN norms quantize their output for the next W8A8
  GEMV in the same launch (the 128-block amax is exact in any order; codes and scales as act_quant_fp8);
  `kv_chain` (kv_norm in place, rope, fp8 fake quant) and `head_rms_rope` (q's per-head RMS and rope) do
  their reference kernels' arithmetic in one launch each. 6 launches fewer per layer, bit-identical: decode
  36.7 / 59.2 / 91.0 -> 35.7 / 58.3 / 89.9 ms at batch 1 / 4 / 8.
- `index_q_chain`: the indexer query's rope, Hadamard and fp4 codes in one launch (CSA layers), the three
  kernels' arithmetic: bit-identical, 35.7 -> 35.6 ms at batch 1. The launches left at decode are mostly
  the MoE dispatch chain and the HC split-K pair, where a fused kernel would serialize work that now runs
  across workgroups.

## 2026-10-15 — tried: two-batch overlap within a prefill step (not adopted)

Splitting a >= 512-row step into two halves and interleaving them per layer (one half's attention on the
card while the other's remote experts run on the APU; MoE half split into begin / end with its own HC
buffers) was bit-identical to the whole-batch step, but slower: 16k prefill 1885 -> 1744 tok/s, the
1024-token prefill 1.37 -> 1.54 s. Each half carries half of every expert's rows, so every expert's weights
are dequantized twice per layer (on the cards and on the APU, whose grouped GEMM pays per tile, not per
row): the APU's work grows more than the overlap hides. At long context the cards seldom wait for the APU
(attention and the indexer dominate). Reverted; `unholy-resident-test --card-margin-bytes` (to keep the
expert placement when a change adds scratch) stays.

## 2026-10-15 — scratch the native path never touches

Each card's activation scratch (3.21 GB in production, planner-counted) held buffers the native-MXFP4 fast
path never reads: three leftovers no code used (`kYF`, `kG`, `kY2F`: 160 MB at 4096 tokens), the EXL3 path's
second transformed input and GEMV partials (`xh3`, `yp`: 603 MB; `yp` was not even in the planner's count),
and the reference expert path's scratch (~200 MB). They are now allocated only when their path can run:
`make_scratch(..., exl3)` / `ApuExperts(..., exl3)` for the EXL3 buffers, `ref_experts_needed()`
(UNHOLY_REF_EXL3 / UNHOLY_REF_MX, or a host-side expert histogram, which `unholy-bench --hist` now sets
before planning) for the reference one. Production plan: scratch 3.21 / 3.34 -> 2.64 / 2.77 GB, hot
experts 1481 / 1393 -> 1524 / 1436, predicted routing on the cards 53.3% -> 54.1%; the ~400 MB of `yp` per
card the planner never counted is now free margin too. At the old placement (`--card-margin-bytes` raised
by the planner's difference) logits are bit-identical.
- The routed experts' w1 and w3 (MXFP4 GEMV and GEMM, cards and APU) run in one launch (grid z picks the
  projection): bit-identical (unholy-mx-test on both architectures, resident checks); decode 35.6 / 89.9 ->
  34.9 / 88.4 ms at batch 1 / 8.
- `moe_dispatch_small`: for steps of <= 32 tokens, moe_count + moe_plan + moe_scatter + remote_tokens +
  stage_remote in one single-block launch (counts, offsets and fill counters in LDS; the same outputs):
  bit-identical, decode 34.9 / 88.4 -> 34.2 / 87.7 ms at batch 1 / 8; unholy-sched-test all ok.
- Native experts with the fp8 remote transport no longer gather their inputs into pair rows: `mx_xform_in`
  reads the tokens through the pair -> token map (`x_rows`), and `kPairIn` (201 MB at 4096 tokens) exists
  only for the EXL3 experts, the bf16 remote transport (UNHOLY_REMOTE_Q=0) and the reference path. The
  indexer's score sub-batch buffer is 64 MB instead of 256 (48 decode rows at 1M context plus their top-k
  scratch fit; prefill takes more, smaller sub-batches). Bit-identical at the old placement; decode 34.2 ->
  33.9 ms at batch 1. Production plan now: scratch 2.24 / 2.36 GB, hot experts 1554 / 1467, routing on the
  cards 54.8% (53.3% before these three changes).

## 2026-10-15 — where the night's work leaves it

`unholy-curve` 1 x 512k (docs/bench/curve-native-1x512k-plx-2026-10-15.txt): decode 34.4 / 36.7 / 38.7 /
42.0 / 47.5 ms per token at 4k / 32k / 128k / 256k / 512k (29.1 -> 21.1 tok/s; the 2026-10-12 curve: 56.7 ...
87.6 ms), prefill 2283 / 1796 / 1472 / 1013 tok/s at 32k / 128k / 256k / 491k, 1246 average (1062).

GSM8K production sweep (serve with the production flags, check-source gsm8k-chat, 48 items, seed 1234,
temperature 1): 29.9 / 55.4 / 68.6 / 82.5 tok/s aggregate at concurrency 1 / 2 / 4 / 8 (2026-10-12: 22.2 /
36.5 / 45.5 / 56.3), median single-stream decode 48.5 tok/s with DSpark (31.0). Flexible match 46 / 46 / 46 /
47 of 48 (2026-10-12: 47 / 47 / 48 / 48): item 2 is wrong in all four runs here, right in the old ones; with
12 seeds both builds get it right 11 times (the same per-request seed lands on another sample when the
logits move by noise). 200 items at concurrency 8: flexible 0.970, strict 0.655 (2026-10-12: 0.980 /
0.645).

Kernels whose arithmetic did not change were checked bit for bit at every step (unholy-resident-test
against the original build at the same expert placement, its batch / verify-chunk checks, combine off);
the numerics that did change are rows of 9-16-row steps (GEMV instead of GEMM: now equal to single-stream
rows), the expert placement (more hot experts) and the DSpark stages' attention (WMMA kernel; same drafts
as the reference).
- The MXFP4 grouped GEMM takes two 16-column tiles per wave on gfx1201 (`mx_gemm_cols`, UNHOLY_MX_GEMM_CW):
  every A fragment read from LDS feeds two WMMAs, each output's k order unchanged. unholy-mx-test, 64
  experts: card 28.4 / 50.9 / 65.1 -> 30.8 / 52.7 / 71.5 TFLOPS at 32 / 96 / 256 rows per expert; on the APU
  (gfx1151) no gain (13.0 / 19.5 / 23.9 vs 13.0 / 19.2 / 23.1), so it keeps one. Its GEMM costs ~31 us per
  expert and projection (the weights' read is ~21 us at its ~210 GB/s) plus ~0.6 us per row. Prefill stays
  bound by the APU. Bit-identical.

End-to-end check of the server (vision on, DSpark off, greedy, logprobs with top 3): the two image
requests of /scratch/vl-test through the original build and through this one with `--card-margin-gb`
raised by exactly the scratch freed (973,602,240 bytes per card, so the plan holds the same 1481 / 1450 hot
experts): every token, logprob and top logprob bit-identical (2 / 2 and 40 / 40). At the production plan
the outputs move as any placement change moves them (the original build with --card-margin-gb 1.1 shows
the same kind of spread). The EXL3 experts' path: bit-identical to the original at its placement too
(unholy-resident-test --exl3 with the margin lowered by the 8,257,536 bytes the planner now counts more).

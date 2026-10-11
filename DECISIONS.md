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

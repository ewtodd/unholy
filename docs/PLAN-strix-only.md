# Plan: unholy on the Strix Halo alone, experts streamed from the SSDs

Date: 2026-11-08. **Status: approved by the owner 2026-11-08 (decisions at the end); P0 started.** Numbers marked
"est." are mine, from the measurements cited; nothing here has been run as a whole.

Goal (owner, 2026-11-08): an unholy mode that uses only the Strix Halo, replacing gufo and its variants
(`gufo-strix` with Qwen3.8-Flash-Next, `gufo-deepseek` with V4.1 IQ1_M), so it runs next to radiance on the two
R9700s. The constraint: at least one of the models unholy serves today must run **at the quantization it is served
at now**. Neither fits the 131 GB of RAM, so whatever does not fit streams from the SSDs, and the work is making
prefill and decode as fast as the APU and the SSDs allow.

## Which model: DeepSeek-V4-Flash native first

Bytes from the checkpoints' safetensors headers (`/scratch/dsv4-ve`, `/scratch/models/glm53-flash-exl3-4.05bpw`):

| | DSV4-Flash native (ds4 mode) | GLM-5.3-Flash EXL3 4.05 (glm mode) |
|---|---:|---:|
| routed experts | 147.17 GB (11,008 x 13.37 MB, MXFP4) | 152.65 GB (12,096 x 12.62 MB, EXL3 4.05) |
| drafter's routed experts | 10.27 GB (DSpark, 768 x 13.37 MB) | 3.63 GB (MTP) |
| everything else | 9.77 GB (dense fp8 6.73, head 1.06, embedding 1.06, vision 0.93) | 9.21 GB (dense 6.74, embedding 1.27, vision 1.73 in EXL3, MTP 0.16) + the release's bf16 tower |
| read per decode token, all resident | 11.24 GB (dense 7.79 + experts 3.45) | 10.98 GB (dense 6.74 + experts 4.24) |
| expert reads per token | 258 (6 x 43) | 336 (8 x 42) |
| routing skew: routed picks caught by the top 60% of experts (aggregate profile, held out) | 78% | 66% |
| same, one 32k code document's own profile (a session's locality, upper bound) | 91-96% (top 30%: 69-84%) | (top 32.5%: 75%) |
| APU kernels today | MXFP4 expert GEMM / GEMV (18.8-26.9 TFLOPS, 212 GB/s) | EXL3 expert GEMM 13.6 TFLOPS, near its structure's limit (DECISIONS 2026-11-03) |
| drafter | DSpark, 5 drafts | MTP, 1 draft (~1.3x) |
| quality | lossless (the release's own MXFP4 + fp8) | KL 1.27e-2 vs FP8 |

DSV4 wins every row that decides streaming: fewer and more skewed expert reads, faster APU kernels, a drafter that
amortizes the 7.8 GB of dense weights over more tokens, and its served precision is the release itself. GLM on the
APU alone would land near prefill ~300-400 tok/s and decode ~12-16 tok/s with MTP (est.: ~20B active parameters on
a 13.6 TFLOPS GEMM; 336 reads per token over a flatter routing). It can follow later on the same expert store
(below), but it is not the first target.

## Memory: what is left for the expert cache

MemTotal on the box is 131.2 GB (`ttm.pages_limit` 124 GiB). Budget for DSV4 native beside radiance:

| item | GB |
|---|---:|
| dense, head, embedding, vision, DSpark's dense part | 10.4 |
| KV pool, 512k tokens (bf16 KV, ~5.9 KB per token) | 3.1 |
| step scratch (8192-token steps) | ~5 |
| streaming stages (one layer's misses, double-buffered) | ~3 |
| unholy host process; prefix snapshots on `/work` with 2 in RAM | ~2.5 |
| radiance's host memory as configured (4.55 GiB pinned; MemAvailable fell 5.2 GiB at its start) / trimmed (section "Radiance") | 5.6 / ~1.8 |
| NixOS, kernel, other services | ~7.5 |
| margin | ~4.3 |
| **expert cache** | **~93 / ~96.5 (59-61% of the 157.4 GB of experts)** |

The anchor: gufo-deepseek fits an 88 GiB (94.5 GB) cache beside radiance as configured and still leaves 7.9-10.3 GiB
available; its fixed part is smaller than DSV4's (V4.1 IQ1_M's non-expert weights are 6.2 GB, no vision tower).

Every GB freed beside it is a GB less to stream on each prefill step (0.13 s) and a few misses fewer in decode,
which is why radiance's RAM matters (section "Radiance").

## The SSDs

`/scratch` (NVMe, WD SN5100, 03.1 x4): 6.2 GB/s, 16 MB random direct reads. `/stream` (3 SATA in RAID 0 on the
ASM1166, 03.2): 1.62 GB/s. Both at once: 7.78 GB/s (fio, 2026-10-23). Neither shares a link with the R9700s (02.5).

- The checkpoint stores each expert as six tensors (w1/w2/w3 and their scales) up to 3.4 GB apart and not 4 KiB
  aligned, and the loader repacks MXFP4 into 16-row chunks (`mx_repack_kernel`). So, two options:
  - **default, no space on `/scratch`:** the NVMe side reads the checkpoint itself (six aligned-superset reads per
    expert into a stage, then the repack kernel into the cache slot: ~0.1 ms per expert on the APU), and `/stream`
    holds a 4 KiB-aligned pack of all 11,776 experts in the loaded layout (157 GB of its 703 GB), read straight
    into slots.
  - **if `/scratch` space is freed** (it is 93% full, 184 GB free): the same pack on the NVMe too. gufo measured
    only ~1% from an expert-contiguous repack, so this is a convenience, not a speedup.
- Reads: `pread` with `O_DIRECT` from ~8 threads, two queues (one per device), each read assigned to the device that
  finishes it first. gufo found io_uring no faster than `pread` at queue depth 4, and io_uring needs the
  service's memlock limit raised.
- The cache lives in pinned host memory: the APU reads pinned host memory at 234 GB/s against 237 for its own
  (DECISIONS 2026-10-11), so `/stream` reads land in their slot with no copy. (gufo's zero-copy reads into
  *managed* memory were slower and reverted; that is a different allocation type, to be checked in P0.)

## Where the time goes

**Decode.** 11.24 GB per token at the APU GEMV's 212 GB/s is ~53 ms: a ceiling of ~19 tok/s without drafting, every
expert resident. Misses add `258 x (1 - hit rate) x 13.37 MB / ~7.5 GB/s`:

| hit rate | misses per token | stall if none hidden |
|---:|---:|---:|
| 95% | 12.9 | 23 ms |
| 97% | 7.7 | 14 ms |
| 98% | 5.2 | 9 ms |
| 99% | 2.6 | 4.6 ms |

The aggregate profiles say ~80% at this cache size, which would be hopeless. The sessions do much better: gufo's
V4.1 store, with 60% of 146 GiB of experts cached, measured 96.3% hits served (88 GiB) and 98.4-99.2% in steady
decode, 91.5% at 44%. antirez's ds4 streaming this very checkpoint (DSV4-Flash-Vision-Exp MXFP4, 145 GiB) on a 128 GB
M5 Max with an 86 GiB cache decodes at 11.9 tok/s fresh and 19.3 once warm (its docs/SSD_STREAMING.md); with that
machine's ~3x the APU's memory bandwidth, 19.3 implies roughly 90% hits under ds4's plainer policy (global LRU on
ROCm, hotness halved every 16 tokens on Metal). So DSV4 at ~60% cached lands somewhere in 90-99%, and where is the
first thing P0 measures. Estimate: **11-18 tok/s plain, 16-27 with DSpark** across that range (consecutive tokens
share few experts, DECISIONS 2026-10-29, so each extra verified row costs ~3.1 GB of experts, ~15 ms, plus its
misses; gufo got 1.57x from DSpark on V4.1 at 99% hits).

**Prefill.** ~26 GFLOP of weights per token, plus the indexer (86,016 x depth FLOPs) and HCA at depth. At ~25 TFLOPS
average (the MXFP4 GEMM reaches 26.9 at 57 rows per expert; a step of 8192 tokens gives ~190) that is ~1.2 ms per
token at shallow depth. A step also reads every cached-out expert it touches, and a large step touches nearly all of
them (from the routing profiles, ~60-62% cached):

| step tokens | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| expert bytes to read (session-like / aggregate) | 26 / 38 GB | 31 / 41 GB | 36 / 44 GB | 40 / 46 GB |
| read time at 7.5 GB/s | 3.5-5.1 s | 4.1-5.5 s | 4.8-5.9 s | 5.3-6.1 s |
| compute | 1.2 s | 2.5 s | 4.9 s | 9.8 s |

With reads overlapped, steps of ~8192 tokens are compute-bound: est. **~500-800 tok/s to 32k, ~400-550 at 128k,
~250-300 at 512k** (the indexer's 45 GFLOP per token at 512k; the range is the gfx1151 dense GEMM's unknown speed
and how much of the reads really hide). ds4 on the M5 Max above: 300 tok/s for an 8k prompt, 263 for a 4k append
(4096-token chunks, only the chunk's selected experts read, no read-ahead for DSV4). The floor is the short step: a
new 1-2k-token turn after a cached prefix still reads ~30-40 GB, **~4-5 s to first token**, whatever the kernels
do. Only more cache or more SSD bandwidth lowers that floor.

For comparison: the three-device ds4 mode does 1837-2194 tok/s prefill and 37-41 tok/s decode; gufo's V4.1 IQ1_M
beside radiance did 890-1111 tok/s and 28-31 tok/s (YoCo halves V4.1's prefill work; DSV4 has no equivalent). So
this mode trades speed for the release's own weights of the model the owner already uses.

## Design

### 1. Expert store

One pool of whole-expert slots (DSV4: one size, 13.37 MB, the drafter's experts included) in pinned host memory,
sized after radiance and the sessions are up (gufo's lesson: a default cache sized at load came out at 4.79 GiB once
the TTM page pool hid memory, so the size is always explicit). Per (layer, expert): slot or none, a score, a pin
count while a step uses it.

Policy, starting from gufo's, which works on the same kind of routing:
- seed from a routing profile (`data/expert_profile_unholy_mixed.txt`, later one built from live traffic via
  `/unholy/routing_profile`);
- score decays with age (`score / (1 + age / 1600)`), each use adds `min(count, 4)`;
- decode and small steps admit every miss; a large prefill step admits only experts with enough rows
  (`count >= max(4, n / 32)`) and streams the rest through a transient stage, so one long document does not flush
  the session's working set;
- never evict an expert the step itself uses;
- after a prefill, the experts its last tokens chose are admitted from the stage (ds4 does this; decode starts
  from them).

The policy is a plain function over routing ids, so P0 can tune it offline on recorded routing traces. Two things
ds4 learned the hard way: slots come from a few large allocations made at start (per-miss allocations cost hundreds
of 12 MB allocations per token), and kernels must read cached experts in place through the per-expert pointer
tables unholy's MoE kernels already take (ds4 copies every selected expert, hits included, into a compact buffer
on every layer: DRAM traffic the APU cannot spare).

### 2. Decode step

Per layer: the router writes its ids to pinned memory; the host plans the layer while the shared expert runs; the
hit experts run at once; the misses are read, and a device-side flag (`hipStreamWaitValue32`, as the cards wait for
the APU today) releases their GEMVs. Outputs do not depend on which experts were hits: every expert's rows are
computed by the same kernel either way, so the mode stays exact.

Hiding misses:
- **hash-routed layers 0-2**: their experts follow from the token id, so their reads start with the step; for
  DSpark, the drafts' ids are known before the verify step, so theirs too (exact, DSV4-specific);
- **next layer from the stale router** (layer L+1's router on layer L's input): gufo's recall 79% one layer ahead;
  it paid only after a layer that waited on reads (predicting every layer was 7% slower);
- DSpark's draft count stays adaptive (its confidence head), with the verify cost per row measured under misses.

### 3. Prefill step

- Steps of 8192 tokens when no decode waits, 2048-4096 when one does (the scheduler's token budget already decides
  step size; it gets the store's read cost as an input).
- While layer L computes, layer L+1's cached-out experts are read: for steps of a few thousand tokens nearly every
  expert is touched, so the stale router's prediction plus a probability threshold covers it. (gufo's router-free
  sweep read 3.6x the demand on V4.1's 2048-row batches; on DSV4's 8192-row steps the demand is ~90% of the sweep;
  P0 checks with real counts.)
- Two waves per layer, as gufo does: the hit experts' gate/up and down while the reads land, then the misses.
- Layer-major super-steps (each cached-out expert read once per 64k tokens instead of once per step) only if the
  8192-token steps leave prefill I/O-bound; gufo shelved it (-32% reads, +4-6 s GPU).

### 4. gfx1151 kernels for what runs on the cards today

The APU already has the MXFP4 experts (GEMM, GLU GEMM, GEMV), the bf16 GEMM, the vision tower, and the reference
kernels. Missing, because they use gfx12-only instructions:

| kernel | today | on gfx1151 |
|---|---|---|
| `w8a8_kernel`, `w8a8_skinny_kernel` (dense fp8 prefill / verify steps) | fp8 WMMA | fp16 WMMA on exactly converted inputs (e4m3 and the activation's e4m3 codes are exact in fp16; the 128x128 block scales stay in fp32) |
| fp8 GEMVs (decode; 70% of decode bytes) | gfx12 conversion instructions | the bit decode today; needs a two-instruction e4m3 -> fp16 decode (shift, multiply by 2^8, exact) to stay at ~212 GB/s |
| `bf16_skinny_kernel` | gfx12 WMMA layout | gfx11 layout |
| sparse attention, HCA attention (`attn_wmma`) | gfx12 WMMA | gfx11 WMMA |
| indexer, prefill (`indexer_fp4_q_kernel`) | gfx12 | fp16 WMMA (e4m3 queries and e2m1 keys are exact in fp16) |

Every new kernel gets a test against its reference kernel. Donors are for structure only: ds4's ROCm backend (MIT)
is its CUDA code through HIP with rocWMMA / hipBLASLt, and runs MXFP4 experts resident only, with Q8 activations
instead of the reference's fp8 (its Strix numbers, Flash IQ2 resident: 207-294 tok/s prefill, 14.7 decode);
gufo's V4.1 kernels (MIT) are fp16 WMMA over IQ1/IQ2 types, no MXFP4. unholy's own APU MXFP4 kernels are already the
fastest of the three.

### 5. One device: model and planner

The resident model and the planner assume two cards plus the APU. The APU-only path is simpler: all 43 layers, the
embedding and the head on one device, no transport and no remote combine; the MoE dispatch calls the store instead
of the APU channel. Checked against the reference forward on the APU (KL at the oracle's noise floor) and against
the three-device service on the same prompts (noise level: the kernels differ).

### 6. Serving

- An instance `unholy-strix` with only the APU visible (`ROCR_VISIBLE_DEVICES`), started after radiance so the
  cache is sized against what radiance really holds; served name `dsv4-flash`.
- Prefix snapshots on the `/work` disk tier (commit cbda888) with 2 in RAM; DSV4's are ~7 MB each.
- `llm-mode`: `qwen` and `dual` become radiance plus `unholy-strix`; `gufo-strix` and `gufo-deepseek` retire
  (e-workstation's `/etc/nixos`, the owner deploys); maitred routes.

### 7. Radiance

Deployed: radiance 1.15.0 (upstream `9ed3ef7`) through `radiance-nix` (`c6a1aef`), service flags
`--prefix-cache-mib 0 --checkpoint-slots 8`. So its RAM prefix tier is already off; what it holds is
- the token embedding, pinned and read over the link: 1.18 GiB per rank, **2.37 GiB** (`--embedding-placement ram`,
  the default);
- 8 checkpoint slots (the linear-attention state at a prefix boundary, which every prefix hit needs): 8 x 153 MiB,
  **1.20 GiB**;
- fixed staging (step ring, media, sampler): **~0.98 GiB**, no flag;
- at load, transiently: six 64 MiB O_DIRECT windows and up to 1.6 GiB of relayout buffers (systemd peaks 7.0-8.1 GB).

Without code: `--embedding-placement vram` (-2.37 GiB of host memory; KV per card 16.93 -> 15.74 GiB, ~510k -> ~475k
tokens; radiance already falls back to it when memory is short) and `--checkpoint-slots 4` (-0.6 GiB) leave ~1.6
GiB pinned. Starting radiance first and sizing unholy's cache after it (gufo-deepseek's unit already waits for
radiance's `/health`) keeps its load peak out of the budget.

The disk tier the owner wants back. Radiance had one up to 1.4.0 (`dc339b4`); 1.5.0 (`b93b531`, 2026-10-13)
removed it: it was write-through behind the RAM tier (it never let RAM shrink), wrote ~60-70 KiB per prefill token
(upstream issue #70: 28 GiB filled in hours) and was not consulted after a restart. On this box it was most likely
never active (the old module passed only `--prefix-cache-dir`, left null). Reverting the removal onto 1.15.0 fails
~80 of 136 hunks; the cache has since gained TTLs, replicated-group slots, mixed block sizes and window snapshots.
So it is a new patch, not a revert:
- **Checkpoint spill (small):** checkpoint slots written to a preallocated O_DIRECT file on `/work` only when evicted
  from RAM, read back on a hit: unholy's own snapshot tier (cbda888) applied to radiance. RAM slots drop to 2 while
  any prefix whose KV pages are still in VRAM (~475k tokens a card) stays resumable. A few hundred lines.
- **KV pages too (large):** the whole prefix on disk, written back on eviction, reloaded after restarts. ~2.5-3.5k
  lines, and #70's write volume has to be designed out first.
- Carried as a patch from the flake: `mkRadiance` takes no patches today; either `radiance-nix` gains a
  `patches ? []` argument or the service module uses `overrideAttrs`. Every radiance bump means rebasing it.

## Milestones

**P0: measure what decides it (1-2 days, one short window).**
1. Routing traces: an `UNHOLY_ROUTING_TRACE` dump (per token: layer, expert ids) from the ds4 path, run on recorded
   real sessions (chat, code, an agent loop, a long document) in a window; an offline simulator replays them through
   the store's policy at 80-110 GB: hit rate, misses per token, read bytes per prefill step. Exit: decode hit rate
   at the planned size. Below ~95% the decode estimate does not hold and the plan comes back to the owner.
2. The reads, as the store will issue them: 13.37 MB `O_DIRECT` reads at queue depths 4-8 from the NVMe and
   `/stream` together, into pinned host memory, with the APU running a bandwidth-bound kernel (and radiance up, if
   possible). Exit: sustained GB/s and per-expert latency.
3. gfx1151 prototypes: the fp8 GEMV with the fast decode (GB/s) and the fp8-as-fp16 GEMM (TFLOPS at DSV4's
   shapes). Exit: the numbers the estimates above assume (>= 200 GB/s, >= 25 TFLOPS).

**Q0: DSV4 native against V4.1 IQ1_M, quality (runs, no engine work).** Both are served today: DSV4 native by the
ds4 mode (the same weights and math the APU mode will run, up to placement-level noise), V4.1 IQ1_M by gufo-deepseek
in the dual mode. The same check-source evals on both, paired by item (docs/compare-protocol.md): AIME, math500,
longctx, gsm8k-chat. The ds4 runs double as P0's routing-trace workload.

**A1: expert store (3-4 days).** Pack tool (`/stream` pack in the loaded layout, checked byte for byte against the
loader's repack), reader threads, pool, policy, metrics (hits, misses, bytes, stall time on `/metrics`). Exit: a
unit test streams every expert through the pool bit-identical to the resident load; the simulator and the store
agree on the P0 traces.

**A2: gfx1151 kernels (4-6 days).** The table in section 4. Exit: each against its reference; GEMVs >= 200 GB/s,
dense GEMM >= 25 TFLOPS.

**A3: one-device model (3-4 days).** Planner and resident model for the APU alone, the experts from A1's store in a
plain blocking mode (no prefetch, no overlap). Exit: KL at the noise floor against the reference forward; logits
identical whatever the cache size (every expert's rows go through the same kernels, hit or miss).

**A4: streaming in the step (4-6 days).** Decode with flags and the prefetches, prefill with the per-layer reads and
two waves, the scheduler's step size. Exit: curves (`unholy-curve`) and served numbers against the estimates.

**A5: serving (2-3 days).** NixOS instance, `llm-mode`, maitred, radiance's patch, a day of real use beside
radiance under load. Exit: no swapping, no OOM, the numbers hold.

**Later, if wanted:** GLM-5.3-Flash on the same store (gfx1151 DSA and attention kernels); V4.1 UD-Q2_K_XL on it
(191 GB of experts, the planned full-device build). The store carries over to a second Strix Halo too, where DSV4
native would no longer need it.

## Faster SSDs, if the floor matters more than the hardware cost

The short-turn floor and the decode misses scale with SSD bandwidth. The 03.2 slot (Gen4 x4, closed) holds the
ASM1166 at Gen3 x2: an NVMe on an M.2-to-x4 adapter there would take the second device from 1.6 to ~6-7 GB/s
(~13 GB/s together; the floor ~4-5 s -> ~2.5-3 s), at the cost of moving `/work` and `/stream` off the SATA drives.
A USB4 NVMe enclosure adds ~3 GB/s but takes a port the two-Strix link would use. Neither is needed to start.

## Decisions (owner, 2026-11-08)

1. DSV4 native first; GLM-5.3-Flash later, on the same store.
2. The expert pack lives on `/stream` only; the NVMe side reads the checkpoint itself.
3. GPU windows as needed for P0.
4. The bar is quality, not a speed number: is full-precision DSV4 better than V4.1 IQ1_M? Measured (Q0 below)
   before the engine work past P0.
5. Radiance: the two flags plus the small checkpoint spill patch to `/work`, applied from the flake.

# DeepSeek-V4.1-Flash-1017 at native MXFP4 with streamed experts: what gufo-deepseek shows

Date: 2026-11-10.
Purpose: the owner's question, after reading gufo-deepseek's served numbers: would a larger V4.1 quant (~Q4) with
experts streamed from the SSDs be excellent on unholy? Notes for the owner to keep in mind; **no decision taken**.
Numbers marked "est." are mine, scaled from the measurements below; nothing here has been run as an unholy mode.

## Summary

- gufo-deepseek streams V4.1 experts far better than our plans assumed: with 63% of the experts cached it hits
  ~99% of routes in decode. The aggregate routing profiles had suggested ~80% at that size.
- "Q4" for V4.1 means the release's own MXFP4 experts: lossless, 288.8 GB, and unholy already has MXFP4 kernels.
- On today's full device (2x R9700 + APU) only ~51% of those experts fit, and the disks give 7.78 GB/s. Est.
  decode ~22-29 tok/s, long-prompt prefill ~800-1500 tok/s, a new 1k-token turn waits ~2.5-6 s: about GLM's speed,
  above gufo IQ2_XXS today, well below DSV4 today. Good, not excellent.
- Disk bandwidth on son-of-anton cannot grow (below). Only memory helps.
- With the R9700s gone, two Strix Halos with an NVMe each just for experts: ~70% cached, 14-28 GB/s of disk, disk
  becomes a minor cost. Speed is then set by the APUs' memory bandwidth and how the two boxes split the work:
  est. ~20-33 tok/s decode and ~1200-2000 tok/s prefill. The top of that range is the excellent case, and it is
  the most engine work.

## What runs today

`/etc/nixos` 40b169af (2026-11-08): gufo-deepseek is gufo plus `ds41-gufo-extension`, serving Unsloth's UD-IQ2_XXS
(146.25 GiB of routed experts, 7.04 GiB dense, Unsloth's IQ4_NL Engram table as shard 6), Lucebox's DSpark drafter,
`--stream-experts --expert-cache-bytes` 92 GiB, one session, 1M context. Before it (2026-11-04/05), ISTA's IQ1_M
(106.7 GiB of experts) with an 88 GiB cache.

## Measured (gufo-deepseek journal, 2026-11-04 to 11-09)

Real use through maitred, one session; the 8-session eval batch of 2026-11-08 is left out. Read from the request
lines (`event=completed`) and the `expert cache interval` lines.

| | IQ1_M, 82.5% of experts cached | UD-IQ2_XXS, 62.9% cached |
|---|---|---|
| requests | 800 (3 runs) | 166 (2 runs) |
| route hits, decode / prefill | 99.8% / 99.2% | 98.5-98.9% / 95.8-97.2% |
| decode, DSpark | 30-36 tok/s (76-85% accepted) | 20-23 tok/s (70-75% accepted) |
| disk read in decode | ~3 MiB/token, 0.7 ms/token waited | 36-46 MiB/token, ~4.9 ms/token waited (~half hidden by prefetch) |
| prefill, 2-64k prompts | 766-942 tok/s | 487-600 tok/s |
| disk read in prefill | 1.2-1.9 MiB/token, ~1.4 GB/s | 5-8 MiB/token, 3.2-4.0 GB/s (NVMe only): disk-bound |
| first token, new turn of <512 / 0.5-2k tokens | 0.9-1.2 s / 1.8-2.0 s | 1.5-1.6 s / 3.7-4.5 s |

How the time is spent: in normal runs prefill takes 7-13% of request time, decode the rest. A typical turn adds
~340-1020 new tokens to a long prefix-cached prompt (prompts up to 446k tokens; only 0.4-1.4% of prompt tokens were
prefilled) and generates ~100-660. Decode speed is what the owner feels most; the wait on a new turn comes second.

## Why "Q4" is native MXFP4

V4.1's checkpoint stores its routed experts as MXFP4 (4.25 bits): 40 layers x 384 experts = 15,360 experts of
18.8 MB, 288.8 GB in all; top-6, so 240 expert reads and 4.51 GB of expert weights per token (DSV4: 3.45 GB).
It is the release itself (KL 0), and unholy's MXFP4 kernels on the cards and the APU (DSV4's) already exist, so
no GGUF kernels are needed (UD-Q2_K_XL needed seven GGUF expert types, `docs/dsv41-flash-quant.md`). An Unsloth
Q4_K build would be larger and still lossy against it.

## Full device as it is today (2x R9700 + APU)

Room for experts ~147 GB (cards ~24.5 GB each, APU ~98 GB), so ~51% of the 288.8 GB stays in memory and ~142 GB
streams. Against gufo IQ2_XXS that is 2.5-4x the bytes to read per turn, with ~1.6-1.9x its disk bandwidth (unholy
reads the NVMe and `/stream` together; gufo got 3.2-4.0 GB/s from the NVMe alone).

| | DSV4 on unholy today | gufo UD-IQ2_XXS today | V4.1 MXFP4, full device (est.) |
|---|---|---|---|
| quality | lossless | KL 0.142 vs Unsloth's Q8 | lossless |
| decode | 37-41 tok/s | 20-23 | ~22-29 |
| prefill, 8-32k prompts | 1837-2194 tok/s | 520-600 | ~800-1500 with 8-32k-token steps (~300-450 with 2k steps) |
| first token, new 1k-token turn | under 1 s | ~4 s | ~2.5-6 s |

The MXFP4 row of `docs/PLAN-glm-dsv41.md`'s D3 table (~235 tok/s prefill, 7-10 tok/s decode) is too pessimistic in
the light of these hit rates. One caveat the gufo numbers do not cover: on the full device ~49 GB of the room is on
the cards; if that part is placed once from a profile instead of following each session, hits come out lower than
gufo's fully adaptive cache.

## Disk bandwidth on son-of-anton is fixed

PCIe: the cards use the M.2 slot (02.5, behind the PEX switch), the NVMe is on 03.1, and the 03.2 slot holds the
ASM1166 with the six SATA drives.

- **A second NVMe in 03.2:** ruled out by the owner (2026-11-10). It would replace the ASM1166 and with it the SATA
  drives (`/stream`, `/work`).
- **More SATA drives in `/stream` instead of `/work`:** no gain. The ASM1166's Gen3 x2 link caps the SATA side.
  Measured 2026-11-10 (fio, read-only, 16 MB random direct reads, 15 s each):

  | | GB/s |
  |---|---:|
  | `/stream` alone (3 drives) | 1.61 |
  | `/work` alone (3 drives) | 1.62 |
  | both at once (all 6 drives) | 1.78 |

  Moving all six drives to `/stream` would take streaming from 7.78 to ~7.95 GB/s (~2%) and cost `/work`, where the
  owner wants the snapshot tier. Capacity is no reason either: `/stream` has 557 GB free, enough for a V4.1 MXFP4
  pack (289 GB) plus the native Engram tables (203 GB) beside the DSV4 pack. (Preallocated files such as
  `radiance-checkpoints.bin` read at ~3.9 GB/s: unwritten blocks come back as zeros without disk I/O. Test with
  written data.)
- `/work` and `/stream` share that link, so writes to `/work` while serving (snapshot or checkpoint spills, on
  eviction only) take a little from `/stream`.

## A second Strix Halo

Room for experts in each configuration. A second box holds only experts: no dense weights, KV cache, drafter or
radiance.

| | room for experts | cached (of 289 GB) | streamed |
|---|---|---|---|
| today: 2x R9700 + 1 APU | ~147 GB | ~51% | ~142 GB |
| 2 APUs, R9700s gone (the card-swap plan) | ~100 + ~105 = ~205 GB | ~70% | ~85 GB |
| 2x R9700 + 2 APUs | ~147 + ~105 = ~250 GB | ~88% | ~40 GB |

70% cached lies between gufo's 63% (~99% decode hits) and IQ1_M's 82% (99.8%). 2x R9700 + 2 APUs is close to
resident (about where the UD-Q2_K_XL plan stood, 41 GB over) and the fastest of the three (speeds not estimated),
but it is four devices across two boxes and keeps the R9700s, which the card-swap plan sends to another host.

## Two Strix Halos, each with its own NVMe for experts

With the R9700s gone, son-of-anton's M.2 slot (02.5, Gen4 x4) is free for an NVMe used only for experts, and the
second box has its own M.2 slots (owner, 2026-11-10).

- **Disk bandwidth:** son-of-anton ~7 GB/s from the new drive, up to ~15 with the SN5100 (6.2) and `/stream` (1.6);
  the second box ~7, or ~14 with both M.2 slots. ~14-28 GB/s together, against 7.78 today. Strix Halo is Gen4 only:
  ~7 GB/s per drive at most, so Gen5 drives buy nothing.
- **What the disks still cost** (~85 GB cold, ~42 GB per box), est.:
  - decode: ~1-2 misses per token, ~1 ms of disk time per token;
  - a new 1k-token turn: ~15 GB to read, ~0.5-1 s (2.5-6 s on today's full device);
  - long prompts: no longer disk-bound.
- **The new limit is the APUs' memory bandwidth.** With no cards to hold them, the dense weights are read by the APUs
  too, at ~212 GB/s each: ~13-14 GB per decode token (native fp8 dense ~9 GB, est. from UD-Q2_K_XL's 8.5 GB at 5-8
  bits; experts 4.5 GB). So decode depends on how the boxes share each token over USB4:

  | split | decode with DSpark (est.) |
  |---|---|
  | by layers (each token runs through both boxes in turn) | ~20 tok/s |
  | each box holds half of every layer's experts; dense weights on both | ~24 tok/s |
  | experts and dense weights both split | ~30-33 tok/s |

- **Prefill:** compute-bound on the two APUs, est. ~1200-2000 tok/s depending on the split and the link. A split by
  halves (layers 0-19 / 20-39) leaves the second box idle in prefill, since with YoCo the second half of the layers
  runs only on the prompt's last 140 rows; a layer split has to interleave.
- **Work:** the top row needs the experts and the dense weights split across two boxes over the USB4 research driver
  (`unholy-card-swap` notes: donnerkeule, ~38-45 Gb/s), plus the gfx1151 kernels of `docs/PLAN-strix-only.md` A2
  (dense fp8 GEMMs and GEMVs, attention). The expert store (A1, A4 there) carries over unchanged.

## All configurations side by side

| | DSV4, full device today | gufo UD-IQ2_XXS, APU | V4.1 MXFP4, full device | V4.1 MXFP4, 2 APUs + NVMe each |
|---|---|---|---|---|
| quality | lossless | KL 0.142 | lossless | lossless |
| decode | 37-41 tok/s | 20-23 | ~22-29 (est.) | ~20 / ~24 / ~30-33 by split (est.) |
| prefill, 8-32k | 1837-2194 tok/s | 520-600 | ~800-1500 (est.) | ~1200-2000 (est.) |
| first token, new 1k turn | under 1 s | ~4 s | ~2.5-6 s (est.) | ~0.5-1 s of disk + compute (est.) |
| hardware | as is | as is | as is | second Strix Halo, two NVMe drives, R9700s out |

## Open checks, when the time comes

1. **Hit rate at 51%:** run gufo-deepseek for a day of normal use with `expertCacheBytes = 75 GiB` (51% of
   UD-IQ2_XXS's experts, the full device's share for MXFP4); multiply its reads by 1.84 (18.8 vs 10.2 MB per
   expert). One line in `hosts/son-of-anton/configuration.nix`, the owner's deploy, `llm-mode dual`.
2. **V4.1's native dense size:** from the checkpoint's safetensors headers (the ~9 GB above is an estimate and the
   softest input to the dual-APU decode numbers).
3. **Card placement on the full device:** hits with the cards' ~49 GB placed from a profile vs following the
   session (the routing-trace simulator of P0.1, once V4.1 routing traces exist).
4. **Two-box split:** which of the three splits the USB4 link and the engine can carry.

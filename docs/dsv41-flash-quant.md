# DeepSeek-V4.1-Flash-1017: which GGUF build for the full-device mode

Date: 2026-11-01.
Purpose: the build unholy should run V4.1-Flash-1017 from on both cards and the APU (the plan's D-milestones,
`docs/PLAN-glm-dsv41.md`). It replaces D2's own EXL3 3.0 quantization run: published GGUF builds at 2-3 bpw now
exist, so unholy needs GGUF kernels rather than a 543.6B-parameter quantization of its own. The APU-only mode (the
gufo fork on ISTA's IQ1_M) is a separate question.

## Decision

**Unsloth's UD-Q2_K_XL** (Dynamic v3.1): 191.1 GB of routed experts, 8.5 GB of everything else, a 135.2 GB Q5_K
Engram file on the SSD. About 41 GB of the coldest experts do not fit the ~150 GB of expert room; they either stream
from the SSD (D3's tier) or are replaced by the same experts from a smaller Unsloth build (below), whichever
measures better. **Fallback: UD-IQ2_XXS** (156.9 GB of experts, ~7 GB over), if neither works.

ISTA-DASLab's GSQ-RCO builds are out: at every size Unsloth's measure better (below). ISTA's IQ1_M stays on the box
for the APU-only test.

## Room for experts

The production plans of 2026-10-29 leave this much for routed experts (cards: free memory minus dense weights, KV,
scratch and a 2 GB margin; APU: free memory minus working buffers, the drafter's experts, the vision tower and a
4 GB margin):

| model | card 0 | card 1 | APU | total |
|---|---:|---:|---:|---:|
| DSV4-VE | 24.56 GB | 24.56 GB | 98.05 GB | 147.2 GB |
| GLM-5.3-Flash | 23.81 GB | 24.58 GB | 104.25 GB | 152.6 GB |

V4.1's non-expert weights in UD-Q2_K_XL (8.48 GB: dense 5.58, shared experts 1.39, embedding / head 1.25, Engram
projections 0.26) are about DSV4's dense share on the cards (8.84 GB), and its drafter is no bigger (Lucebox
Q2_K/Q4_K 4.9 GB, Unsloth BF16 8.5 GB, against DSV4's 10.3 GB of drafter experts on the APU). Compressed KV is small.
So **~150 GB of experts**. The Engram tables stay on the SSD in every build: a token reads 2 x 24 rows.

## Candidates

Bytes from the published headers (`tools/ref/gguf_inventory.py`); quality from Unsloth's page (2026-10-30, their
measurements). Bytes per token: 6 of 384 experts, 1.5625% of the routed bytes (DSV4 today: 3.45 GB).

| build | routed experts | other | past ~150 GB | 6-bench avg | KL vs F32, long17k / top-1 | experts per token |
|---|---:|---:|---:|---:|---:|---:|
| ISTA IQ2_XXS | 132.07 GB | 6.16 GB | 0 | 83.68 | - | 2.06 GB |
| UD-IQ1_M | 132.21 GB | 7.59 GB | 0 | 85.55 | 0.241 / 88.2% | 2.07 GB |
| ISTA IQ2_XS | 149.06 GB | 6.16 GB | 0 (no slack) | 85.64 | - | 2.33 GB |
| UD-IQ2_XXS | 156.88 GB | 8.42 GB | ~7 GB | 86.63 | 0.142 / 91.2% | 2.45 GB |
| **UD-Q2_K_XL** | **191.14 GB** | **8.48 GB** | **~41 GB** | **87.49** | **0.088 / 93.3%** | **2.99 GB** |
| UD-IQ3_XXS | 224.30 GB | 9.15 GB | ~74 GB | - | 0.043 / 95.1% | 3.50 GB |
| UD-IQ3_S | 253.35 GB | 8.99 GB | ~103 GB | - | 0.020 / 96.8% | 3.96 GB |

The 6-benchmark average is AIME25, HMMT Feb26, GPQA-Diamond, MMLU-Pro, IFEval and LiveCodeBench v6 at reasoning
effort 50; Unsloth's UD-Q8_K_XL reference scores 87.68. UD-Q2_K_XL is at the reference within noise, and its KL is a
third lower than UD-IQ2_XXS's, a difference the benchmarks (AIME25: +-5-7 points at 95% over 30 questions)
cannot show. Every candidate but UD-IQ3_XXS / IQ3_S reads fewer expert bytes per token than DSV4 does now.

Expert types per build (the dense parts use Q5_K / Q6_K / Q8_0, BF16 and F32):

- UD-Q2_K_XL: IQ3_XXS 86.7, IQ2_XXS 33.9, IQ2_S 29.0, IQ3_S 19.5, MXFP4 16.8, IQ2_XS 5.2 GB.
- UD-IQ2_XXS: IQ2_S 65.3, IQ2_XXS 38.5, IQ3_XXS 26.0, IQ1_S 17.7, IQ2_XS 7.9, Q2_K 1.5 GB.
- UD-IQ1_M: IQ2_XXS 59.6, IQ1_S 39.8, IQ2_XS 18.3, IQ2_S 14.5 GB.
- ISTA IQ2_XS: IQ2_XXS 88.8, IQ2_XS 35.4, IQ2_S 23.2, IQ3_XXS 1.7 GB.

## ISTA's numbers

ISTA's card reports its builds at 98.7% (IQ1_M), 100.0% (IQ2_XXS) and 99.8% (IQ2_XS) of the base model's average
over AIME25, GPQA-Diamond and LiveCodeBench v6, with the base and all three builds at 100.00 on AIME25. Unsloth
re-measured the same files with more samples (AIME25 k=16, GPQA-Diamond k=8, LiveCodeBench k=3, temperature 1,
effort 50) against its UD-Q8_K_XL: 86.61 / 87.95 / 89.53 against 91.15, i.e. 95.0 / 96.5 / 98.2%. The ceiling
scores (the base itself at 100 on AIME25, two builds at exactly the base's 89.71 on LiveCodeBench) point to a small
or few-sample evaluation, not to invented numbers; the GPQA drop (90.5 -> 85.1 for IQ1_M) and ~3-point LiveCodeBench
drops are the losses. At equal size Unsloth's builds measure better on KL against its Q8 over 7 text sets (0.12 vs
0.215 at ~232 GB, 0.09 vs 0.166 at ~250 GB) and on the 6-benchmark average (83.53 vs 81.78, 85.55 vs 83.68).
Unsloth competes with ISTA and measures against its own Q8, so this is one side's evidence: the oracle (below)
checks it.

## What follows for the engine

- **GGUF kernels:** expert types IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, Q2_K (the fallback's), MXFP4
  (exists); dense Q5_K, Q6_K, Q8_0; the Engram rows at Q5_K (IQ4_NL for the 1-bit builds). GGUF keeps a layer's
  experts as one tensor of one type, each expert a contiguous slice, so experts can come from different builds.
- **The ~41 GB:** two ways, to measure against each other once the model runs:
  - D3's SSD tier (NVMe 6.2 GB/s + `/stream` 1.6 GB/s). Decode depends on how often the cold experts are routed
    (the gufo fork's IQ1_M, 121 GB, on one Strix Halo: 24.4 tok/s resident, 17.9 with a 64 GiB expert cache and the
    rest read from the SSD; prefill 1212 -> 752 tok/s); prefill reads every non-resident expert each large step:
    41 GB at 7.8 GB/s caps 4k-token steps near ~770 tok/s, larger steps less so.
  - A resident mix: the hot experts from UD-Q2_K_XL, the coldest from UD-IQ2_XXS or UD-IQ1_M, chosen with a V4.1
    routing profile (which does not exist yet).
- **Quality check of our own:** the D4 oracle (DeepSeek's reference code on the native checkpoint) gives the KL of
  UD-Q2_K_XL and the fallback against the full model, independent of both quantizers.

## On disk (son-of-anton)

- `/scratch/models/dsv41-flash-1017-unsloth-gguf/UD-Q2_K_XL/`: the 7 shards, 334.8 GB (download started 2026-11-01).
- `/scratch/models/dsv41-flash-1017-gsq-rco-gguf/`: ISTA IQ1_M (shards 1-5, shard 6 linked to the IQ4_NL Engram
  file), the Engram file (110.6 GB), the image projector, Lucebox's DSpark drafter (Q2_K/Q4_K). ISTA's IQ2_XS was
  deleted on 2026-11-01.

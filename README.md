# unholy
<!---->
A from-scratch serving engine for **DeepSeek-V4-Flash** and **GLM-5.3-Flash** on a single desktop machine.
It runs them across two AMD Radeon AI PRO R9700 cards and the Radeon 8060S iGPU of a Ryzen AI Max+ 395
("Strix Halo"): DeepSeek-V4-Flash at native precision (MXFP4 routed experts, fp8 dense weights, bf16 KV),
GLM-5.3-Flash as an EXL3 4.05-bit quant.
It serves an
OpenAI-compatible API with vLLM-style continuous batching, prefix caching, speculative decoding with
each checkpoint's own drafter, image input (DeepSeek-V4-Flash), and contexts up to 1M tokens.
<!---->
C++20 and HIP; no Python at runtime; every kernel is our own (no rocBLAS / hipBLASLt / CK). It builds
with Nix against TheRock's ROCm 10 SDK.
<!---->
It is *vertical*: a couple of models, one machine.
It was built in one day by Claude Opus 5.5 (RIP software engineering), with an engineering log
(`DECISIONS.md`) that records every measurement and dead end along the way.
<!---->
## The machine
<!---->
```
                Ryzen AI Max+ 395, 128 GB LPDDR5X
          ┌─────────────────────────────────────────────┐
          │  CPU: tokenizer, scheduler, HTTP server       │
          │  Radeon 8060S iGPU (gfx1151, ~124 GB GTT):    │
          │    ~100-120 GB of "cold" routed experts,      │
          │    drafter experts, vision tower              │
          └──────────────┬──────────────────────────────┘
                         │ PCIe Gen4 x4 (M.2), shared
                ┌────────┴────────┐
                │ PEX88048 switch │  card <-> card at 26 GB/s
                └───┬─────────┬───┘
            Gen4 x16│         │Gen4 x16
          ┌─────────┴──┐   ┌──┴─────────┐
          │ R9700 #0   │   │ R9700 #1   │  gfx1201, 32 GB each
          │ layers 0-21│   │layers 22-42│  dense weights, KV, "hot" experts,
          │ + embedding│   │ + head     │  drafter (dense, card 1)
          └────────────┘   └────────────┘
```
<!---->
DeepSeek-V4-Flash (43 layers, 256 routed experts with 6 active per token, hyper-connections, compressed
sparse attention with an fp4 indexer, hash-routed layers, a 3-stage DSpark drafter, a 32-layer ViT) is
about 168 GB on disk.
GLM-5.3-Flash (45 layers: 34 of Kimi delta attention, a linear recurrence, and 11 of DSA, sparse MLA over the
2048 positions its indexer selects; 288 routed experts with 8 active; hyper-connections; one MTP layer) is
165 GB as the EXL3 4.05 quant (`docs/glm53-flash-quant.md` for why that build).
Either is more than the 64 GB of VRAM, so the planner splits the layers between
the two cards (the diagram shows DeepSeek-V4-Flash; GLM-5.3-Flash splits at layer 23).
Each layer's dense weights, caches and its most-used ("hot") experts live on its card;
the hot set is chosen from routing histograms measured on real traffic, and fills the VRAM left after
the dense weights and KV. The rest of the experts live on the iGPU and are computed there, with
device-side flags between the cards and the iGPU so the host never waits. Large prefill steps run as two
overlapping halves, so the iGPU computes one half's experts while the cards compute the other half's
attention.
<!---->
## Performance
<!---->
Measured on that machine, behind the PCIe switch (October 2026). Both models through `unholy serve` with
their production settings, measured the same way (`docs/bench/compare-perf-dsv4-glm-2026-10-24.txt`):
<!---->
| | DeepSeek-V4-Flash | GLM-5.3-Flash |
|---|---|---|
| prefill, one request, unique text (time to first token) | 1681 tok/s at 8k, 1752 at 32k, 1763 at 128k, 1742 at 512k (295 s) | 1037 tok/s at 8k, 1249 at 32k, 1425 at 128k, 1476 at 512k (355 s) |
| decode, one stream, production drafting | 33-36 tok/s to 128k, 30-32 at 256-512k (DSpark, 5 drafts, 55-86% accepted) | 24-25 tok/s from 8k to 512k (MTP, 1 draft, 75-94% accepted) |
| GSM8K chat, engine-batch, 1 / 2 / 4 / 8 requests | 37.9 / 48.2 / 55.6 / 64.2 tok/s | 25.8 / 32.1 / 36.5 / 50.2 tok/s |
<!---->
DeepSeek-V4-Flash's own curves (`unholy-curve`, one sequence in 4096-token chunks; before the two-part prefill
steps, which add 17% at 64k): prefill 2463 tok/s at 32k depth, 2012 at 128k, 1702 at 256k, 1212 at 491k,
1406 averaged over 0-512k; decode without speculation 29.4 tok/s at 4k depth, 22.4 at 512k. Its check-source
GSM8K chat sweep: 31.6 / 54.3 / 69.9 / 85.5 tok/s aggregate at 1 / 2 / 4 / 8 concurrent requests.
<!---->
For DeepSeek-V4-Flash, `unholy serve` trades bit-identical outputs for speed by default: more hot experts on
the cards (the scratch the native path never touches), the experts' GEMVs for verify steps of up to 64 tokens
and the drafter's attention on the WMMA kernel; outputs move at the noise level, like any change of expert
placement.
`--exact 1` gives the 2026-10-12 build's outputs for the same steps instead -- token for token under
concurrency too when the requests reach the engine alike (`unholy-engine-batch`) -- at 31.2 / 51.8 / 62.4 /
72.1 tok/s on the same sweep. Prefill and single-stream decode are the same either way. (DECISIONS 2026-10-15)
<!---->
Production settings: DeepSeek-V4-Flash with a 1.5M-token KV pool shared by up to 8 requests of up to 768k
tokens each, DSpark with 5 drafts, 64 prefix-cache snapshots, the vision tower loaded; GLM-5.3-Flash with a
1M-token pool, 3072-token steps, MTP drafts and the snapshots that fit (about 39). Depth curves are in
`docs/bench/`.
<!---->
## Accuracy
<!---->
Correctness is checked against DeepSeek's own inference code (`inference/` in the checkpoint), run
layer by layer on one card as an oracle (`tools/ref/oracle`).
Two checks:
<!---->
- **Against the reference.** On a 1032-position code prompt, the native path's mean KL to the oracle is
  9.6e-3, with 95.7% top-1 agreement. That is the reference's own noise floor: reordering the reference's
  operations gives 9.9e-3 and 96.2%. The production transport, which sums the iGPU's experts in bf16, is
  at 1.0e-2.
  GSM8K (chat, 200 questions) scores 98% (flexible match).
- **Bit for bit with itself.** Chunk boundaries, batch composition, step pipelining and expert placement
  change nothing: the logits are bit-identical across all of them.
  The one exception is the bf16 sum of
  the iGPU's experts; `UNHOLY_REMOTE_COMBINE=0` turns it off for these checks.
<!---->
GLM-5.3-Flash is checked against the transformers implementation running Z.AI's FP8 release
(`tools/ref/glm_oracle`). Our reference forward of that release is at the oracle's noise floor (KL 3.1e-3,
98.6% top-1 on code; reordering the oracle's attention gives 3.4e-3). The EXL3 4.05 build served here is at
1.27e-2 / 96.7% on code and 7.1e-2 / 93.2% on GSM8K chat turns, and the fast path keeps that (1.33e-2). Its
outputs do not depend on chunking, the two-part steps or speculative drafts (a rejected draft
rolls the recurrent state back exactly); the iGPU's experts come back as one fp32 sum per token, so like
DeepSeek-V4-Flash's they depend on the expert placement at the noise level.
<!---->
Every optimized kernel has a reference kernel next to it (`src/kernels/ref_*.hip`), and its test compares
against that reference: bit-identical where the math allows, ulp-level where it doesn't (`tools/kernel_test`).
<!---->
## Features
<!---->
- **OpenAI-compatible server** (`unholy serve`):
  - endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`, `/detokenize`,
    `/health`, `/metrics` (Prometheus);
  - request features: SSE streaming with keepalives, tool calls, reasoning (`reasoning` and
    `reasoning_content`; `reasoning_effort` none/low/high/max, and low/high/max for GLM-5.3-Flash, which
    always thinks), stop strings, logprobs, usage with `cached_tokens`;
  - optional API key (`--api-key` or `UNHOLY_API_KEY`).
- **The chat formats and tokenizers, matching the references exactly**: a PCRE2 BPE tokenizer
  identical to Hugging Face's; the DSV4 template with its streaming tool-call / reasoning parser,
  checked against the reference on thousands of generated conversations; GLM-5.3-Flash's template and a
  parser following vLLM's GLM-4.7 one.
- **Scheduling as in vLLM**, configured the same way: one KV pool of `--kv-tokens` positions in
  256-token pages, `--max-model-len`, `--max-num-seqs`.
  - Continuous batching with chunked prefill and recompute preemption.
  - Decodes always go first. Prefill chunks are packed shortest-remaining first, so a chat continuation
    is not queued behind a long new prompt; `--prefill-order fcfs` restores arrival order, and vLLM's
    `--long-prefill-token-threshold` is available.
  - Two steps in flight, so the cards and the iGPU overlap.
- **Prefix caching**: pages named by chained token hashes and refcounted, with LRU eviction.
  Both models
  keep per-sequence state outside the paged cache (DSV4: the sliding window and compressor rings; GLM: the
  delta-attention recurrent states, ~148 MB a sequence).
  That
  state is snapshotted to pinned host memory at page boundaries, so a cached prefix is resumed exactly.
- **Speculative decoding** with each checkpoint's drafter: DeepSeek-V4-Flash's 3-stage DSpark block drafter,
  its drafts filtered by its confidence head; GLM-5.3-Flash's MTP layer, one draft per step, with the
  recurrent state rolled back when the draft is rejected. Drafts are verified with exact speculative
  sampling, so outputs are distributed exactly as without them.
- **Vision**: images in chat messages (data URLs; JPEG, PNG, WebP), preprocessed on the CPU bit-exactly
  as the reference does, and encoded by the ViT on the iGPU in under a second.
- **Contexts to 1M tokens**: DeepSeek-V4-Flash with bf16 KV only, by design; GLM-5.3-Flash keeps its DSA
  latent in fp8 (e4m3, a power-of-two scale per 64 values).
- **Placement planner** (`unholy plan`): what goes where for a given KV geometry, from the checkpoint and
  real allocation sizes. Routing histograms of live traffic are exported at `GET /unholy/routing_profile`
  for re-planning, and for GLM-5.3-Flash `GET /unholy/routing_share` gives the share served by the cards'
  experts; `tools/box/mklong.py`, `routing-depth.sh` and `mixprofile.py` build profiles from broad traffic
  (`data/expert_profile_glm53_broad.txt`).
- **Two-part prefill steps**: steps of 2048 tokens or more run as two halves software-pipelined on each
  card, so the iGPU's experts for one half overlap the cards' attention for the other; outputs are
  bit-identical either way (`UNHOLY_DSV4_SPLIT=0` / `UNHOLY_GLM_SPLIT=0` turn it off).
- **Logs**: one line per request (queue time, time to first token, prefill and decode tok/s, draft
  acceptance), and a periodic engine line (throughput, queue, KV use, prefix-cache hit rate, draft
  acceptance by position).
- **EXL3 3-bit experts** (`--exl3`), as an alternative to the native ones.
<!---->
## Building
<!---->
You need an x86-64 Linux machine with Nix (flakes enabled). The toolchain is TheRock's ROCm 10 SDK,
packaged by [`therock-sdk-nix`](https://github.com/ewtodd/therock-sdk-nix).
The build produces fat
binaries for gfx1201 and gfx1151.
<!---->
```sh
nix build github:ewtodd/unholy          # result/bin/unholy and the test tools
```
<!---->
For development:
<!---->
```sh
nix develop -c bash -c 'cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=$CMAKE_HIP_COMPILER \
  -DCMAKE_HIP_COMPILER=$CMAKE_HIP_COMPILER -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH \
  -DCMAKE_PREFIX_PATH=$ROCM_PATH && ninja -C build'
```
<!---->
Other flakes can build it against their own nixpkgs with `lib.x86_64-linux.mkUnholy { pkgs = ...; }`.
<!---->
## Running
<!---->
```sh
unholy serve --model /path/to/DeepSeek-V4-Flash \
  --kv-tokens 1572864 --max-model-len 524288 --max-num-seqs 8 --max-batch-tokens 4096 \
  --prefix-snapshots 64 --spec-drafts 5 \
  --served-model-name dsv4-flash --host 0.0.0.0 --port 8200
```
<!---->
GLM-5.3-Flash (the EXL3 quant directory; `unholy serve` picks the engine from the checkpoint):
<!---->
```sh
unholy serve --model /path/to/glm53-flash-exl3-4.05bpw \
  --kv-tokens 1048576 --max-model-len 786432 --max-num-seqs 8 --max-batch-tokens 3072 \
  --prefix-snapshots 64 --spec-drafts 1 \
  --served-model-name glm53 --host 0.0.0.0 --port 8250
```
<!---->
```sh
curl http://localhost:8200/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dsv4-flash",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "reasoning_effort": "high",
  "stream": true
}'
```
<!---->
Loading takes about 35 s.
`unholy plan` with the same flags prints the placement without
loading anything.
`unholy serve` with no arguments lists every flag; the comment at the top of
`tools/unholy/main.cpp` explains them.
<!---->
`unholy-probe` measures what the machine's PCIe paths actually do: bandwidth and signalling latency
between every pair of devices, with every transfer method verified. It's the first thing to run on
new hardware.
<!---->
## Hardware assumptions
<!---->
This is not a general-purpose engine:
- The device layout is two gfx1201 cards for the layers plus one large-memory gfx1151 iGPU for the
  cold experts.
  The planner sizes everything from what the devices report, but the split itself is
  fixed.
- It needs roughly 100-120 GB of iGPU-visible memory (GTT) beyond the cards' 64 GB.
- Transfers follow the rules measured on this platform (`DECISIONS.md`, 2026-10-11): the reading side
  pulls, and the iGPU only touches its own memory and pinned host memory. Some peer paths that the
  runtime reports as available are broken on this machine, and the engine avoids them.
- Card-to-card reads behind a PCIe switch need ACS redirect off
  (`pci=disable_acs_redir=<switch vendor:device>` on the kernel command line). Without it they hang
  at the root complex.
<!---->
## Repository
<!---->
| | |
|---|---|
| `src/engine` | scheduler and paged cache (`engine`), placement planner, the resident multi-device model, sampler |
| `src/model` | checkpoint loading, the DSV4 forward pass, the vision tower, GLM-5.3-Flash (`glm5*`: reference and fast paths) |
| `src/kernels` | HIP kernels: WMMA GEMMs (fp8, bf16, MXFP4, EXL3), sparse attention, the fp4 indexer, hyper-connections, MoE dispatch, DSpark, ViT, GLM's delta attention and DSA (`glm_fast`); `ref_*` are the reference kernels |
| `src/server`, `src/text`, `src/vision` | HTTP/OpenAI layer, tokenizer and chat format, image preprocessing |
| `tools/` | `unholy`, `unholy-probe`, benchmarks (`unholy-bench`, `unholy-curve`), tests, the reference/oracle scripts (`tools/ref`), and the scripts for the dev box (`tools/box`) and the model comparison (`tools/compare`) |
| `data/` | expert routing profiles for the planner |
| `DESIGN.md` | the target and the shape of the system |
| `DECISIONS.md` | dated decisions with their measurements: why each piece is the way it is |
| `STATUS.md` | what's done and what's open |
<!---->
## Third-party code and credits
<!---->
Code taken from other projects, with its license:
<!---->
- `third_party/gufo/json.hpp`: from [gufo](https://github.com/gufo-org/gufo), MIT License,
  Copyright (c) 2026 gufo contributors (`third_party/gufo/LICENSE`).
- `src/vision/image.cpp` (`resize_bicubic` and its coefficient tables): a port of Pillow's
  `libImaging/Resample.c` ([Pillow](https://github.com/python-pillow/Pillow)), MIT-CMU License,
  Copyright (c) 1997-2011 by Secret Labs AB, Copyright (c) 1995-2011 by Fredrik Lundh and contributors,
  Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors (`third_party/pillow/LICENSE`).
- `src/model/exl3.*`: the EXL3 format (trellis layout, codebooks, tensor-core permutation) follows
  [exllamav3](https://github.com/turboderp-org/exllamav3), MIT License, Copyright (c) 2025 Turboderp.
- `data/expert_profile_dsv4engine_code32k.txt`: a routing histogram from dsv4-engine (Apache-2.0).
- GLM-5.3-Flash's EXL3 quants are turboderp's (exllamav3, MIT); the GLM output parser follows vLLM's
  GLM-4.7 parser (Apache-2.0) without copying code.
<!---->
The model's math follows DeepSeek's reference inference code (`inference/` in the DeepSeek-V4-Flash
checkpoint).
The oracle scripts in `tools/ref` import it from the checkpoint at run time; none of it
is copied here.
DeepSeek-V4-Flash's weights and reference code are DeepSeek's, under their own license.
GLM-5.3-Flash's math follows the transformers implementation and Z.AI's release; its weights are Z.AI's,
under their own license.
<!---->
The design also draws on these projects, without copying code: gufo (MIT), antirez ds4.c (MIT),
exllamav3 (MIT), Strata (MIT), llama.cpp / ggml and the llama-halo-hybrid ds4-hybrid branch (MIT),
vLLM and sglang (Apache-2.0), radiance libr4d (Apache-2.0), dsv4-engine (Apache-2.0).
<!---->
## License
<!---->
Copyright 2026 Ethan Todd.
Licensed under the Apache License, Version 2.0 (`LICENSE`).

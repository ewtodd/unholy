# unholy
<!---->
A from-scratch serving engine for **DeepSeek-V4-Flash** on a single desktop machine.
It runs the
native-precision model (MXFP4 routed experts, fp8 dense weights, bf16 KV) across two AMD Radeon AI PRO
R9700 cards and the Radeon 8060S iGPU of a Ryzen AI Max+ 395 ("Strix Halo").
It serves an
OpenAI-compatible API with vLLM-style continuous batching, prefix caching, speculative decoding with
the checkpoint's own drafter, image input, and contexts up to 1M tokens.
<!---->
C++20 and HIP; no Python at runtime; every kernel is our own (no rocBLAS / hipBLASLt / CK). It builds
with Nix against TheRock's ROCm 10 SDK.
<!---->
It is *vertical*: one model, one machine.
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
          │    ~100 GB of "cold" routed experts,          │
          │    DSpark drafter experts, vision tower       │
          └──────────────┬──────────────────────────────┘
                         │ PCIe Gen4 x4 (M.2), shared
                ┌────────┴────────┐
                │ PEX88048 switch │  card <-> card at 26 GB/s
                └───┬─────────┬───┘
            Gen4 x16│         │Gen4 x16
          ┌─────────┴──┐   ┌──┴─────────┐
          │ R9700 #0   │   │ R9700 #1   │  gfx1201, 32 GB each
          │ layers 0-21│   │layers 22-42│  dense weights, KV, "hot" experts,
          │ + embedding│   │ + head     │  DSpark drafter (dense, card 1)
          └────────────┘   └────────────┘
```
<!---->
The model (43 layers, 256 routed experts with 6 active per token, hyper-connections, compressed
sparse attention with an fp4 indexer, hash-routed layers, a 3-stage DSpark drafter, a 32-layer ViT) is
about 168 GB on disk.
That is more than the 64 GB of VRAM, so the planner splits the layers between
the two cards.
Each layer's dense weights, caches and its most-used ("hot") experts live on its card;
the hot set is chosen from routing histograms measured on real traffic, and fills the VRAM left after
the dense weights and KV. The rest of the experts live on the iGPU and are computed there, with
device-side flags between the cards and the iGPU so the host never waits.
<!---->
## Performance
<!---->
Measured on that machine, behind the PCIe switch (October 2026):
<!---->
| | |
|---|---|
| prefill, one sequence, 4096-token chunks | 1861 tok/s at 32k depth, 1525 at 128k, 1092 at 256k, 896 at 491k; **1062 tok/s averaged over 0-512k** |
| decode, one sequence, no speculation | 17.6 tok/s at 4k depth (56.7 ms/token), 11.4 tok/s at 512k |
| decode with DSpark (5 drafts) | ~31 tok/s per stream on chat traffic; 70-90% of drafts accepted |
| GSM8K chat sweep, production settings | 22.2 / 36.5 / 45.5 / 56.3 tok/s aggregate at 1 / 2 / 4 / 8 concurrent requests |
<!---->
Production settings: a 1.5M-token KV pool shared by up to 8 requests of up to 512k tokens each,
DSpark with 5 drafts, 64 prefix-cache snapshots, the vision tower loaded. Depth curves are in
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
Every optimized kernel has a reference kernel next to it (`src/kernels/ref_*.hip`), and its test compares
against that reference: bit-identical where the math allows, ulp-level where it doesn't (`tools/kernel_test`).
<!---->
## Features
<!---->
- **OpenAI-compatible server** (`unholy serve`):
  - endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`, `/detokenize`,
    `/health`, `/metrics` (Prometheus);
  - request features: SSE streaming with keepalives, tool calls, reasoning (`reasoning` and
    `reasoning_content`; `reasoning_effort` none/low/high/max), stop strings, logprobs, usage with
    `cached_tokens`;
  - optional API key (`--api-key` or `UNHOLY_API_KEY`).
- **The DeepSeek chat format and tokenizer, matching the reference exactly**: a PCRE2 BPE tokenizer
  identical to Hugging Face's, and the DSV4 template with its streaming tool-call / reasoning parser,
  checked against the reference on thousands of generated conversations.
- **Scheduling as in vLLM**, configured the same way: one KV pool of `--kv-tokens` positions in
  256-token pages, `--max-model-len`, `--max-num-seqs`.
  - Continuous batching with chunked prefill and recompute preemption.
  - Decodes always go first. Prefill chunks are packed shortest-remaining first, so a chat continuation
    is not queued behind a long new prompt; `--prefill-order fcfs` restores arrival order, and vLLM's
    `--long-prefill-token-threshold` is available.
  - Two steps in flight, so the cards and the iGPU overlap.
- **Prefix caching**: pages named by chained token hashes and refcounted, with LRU eviction.
DSV4 also
  keeps per-sequence state outside the paged cache (the sliding window, the compressor rings).
  That
  state is snapshotted to pinned host memory at page boundaries, so a cached prefix is resumed exactly.
- **DSpark speculative decoding** with the checkpoint's 3-stage block drafter. Drafts are filtered by
  its confidence head and verified with exact speculative sampling, so outputs are distributed
  exactly as without it.
- **Vision**: images in chat messages (data URLs; JPEG, PNG, WebP), preprocessed on the CPU bit-exactly
  as the reference does, and encoded by the ViT on the iGPU in under a second.
- **Contexts to 1M tokens**, with bf16 KV only, by design.
- **Placement planner** (`unholy plan`): what goes where for a given KV geometry, from the checkpoint and
  real allocation sizes. Routing histograms of live traffic are exported at `GET /unholy/routing_profile`
  for re-planning.
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
```sh
curl http://localhost:8200/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dsv4-flash",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "reasoning_effort": "high",
  "stream": true
}'
```
<!---->
Loading the model takes about 35 s.
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
- It needs roughly 100 GB of iGPU-visible memory (GTT) beyond the cards' 64 GB.
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
| `src/model` | checkpoint loading, the DSV4 forward pass, the vision tower |
| `src/kernels` | HIP kernels: WMMA GEMMs (fp8, bf16, MXFP4, EXL3), sparse attention, the fp4 indexer, hyper-connections, MoE dispatch, DSpark, ViT; `ref_*` are the reference kernels |
| `src/server`, `src/text`, `src/vision` | HTTP/OpenAI layer, tokenizer and chat format, image preprocessing |
| `tools/` | `unholy`, `unholy-probe`, benchmarks (`unholy-bench`, `unholy-curve`), tests, and the reference/oracle scripts (`tools/ref`) |
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
<!---->
The model's math follows DeepSeek's reference inference code (`inference/` in the DeepSeek-V4-Flash
checkpoint).
The oracle scripts in `tools/ref` import it from the checkpoint at run time; none of it
is copied here.
DeepSeek-V4-Flash's weights and reference code are DeepSeek's, under their own license.
<!---->
The design also draws on these projects, without copying code: gufo (MIT), antirez ds4.c (MIT),
exllamav3 (MIT), Strata (MIT), llama.cpp / ggml and the llama-halo-hybrid ds4-hybrid branch (MIT),
vLLM and sglang (Apache-2.0), radiance libr4d (Apache-2.0), dsv4-engine (Apache-2.0).
<!---->
## License
<!---->
Copyright 2026 Ethan Todd.
Licensed under the Apache License, Version 2.0 (`LICENSE`).

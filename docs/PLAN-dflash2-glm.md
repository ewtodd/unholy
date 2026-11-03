# Plan: DFlash2 speculative decoding for GLM-5.3-Flash

Date: 2026-11-03. The owner's decision, against DECISIONS 2026-10-29 (which judged it not worth building). The first
milestone measures what decides it, DFlash2's acceptance on our GLM, before any engine work.

**Status: stopped after F1 (owner, 2026-11-03).** On our GLM the drafter lands 1.74 / 2.17 / 2.54 tokens per step at
widths 2 / 3 / 8, at best 1.15x plain decode against MTP's ~1.3x (DECISIONS 2026-11-03). The cost is the box, not
only the drafter: every verified token streams ~2 GB of cold experts on the APU (~12.5 ms), so even Qwen-level
acceptance would only reach ~1.3x. DFlash2 pays where verifying 8 tokens costs about as much as 1, i.e. dense
models such as radiance's Qwen3.8-27B. What F1 left behind: the hidden-state capture (`glm5_set_feature_capture`),
`unholy-glm-features`, `tools/box/mkdflash-prompts.py` and `tools/ref/dflash2_ref.py`.

## What DFlash2 is

`incoai/GLM-5.3-Flash-DFlash2` (CC BY-NC-ND 4.0: local use only, no published derivatives), revision `bf582e4`, on
son-of-anton at `/scratch/models/glm53-flash-dflash2/`: a block-diffusion drafter, 81 bf16 tensors, 2.34 GB.

- **Context:** for every committed token, the target's residual stream after layers 5, 14, 24, 33 and 42 (GLM:
  the mean of the 4 hyper-connection streams, SGLang PR #36708), concatenated (20480) -> `fc` -> `hidden_norm`
  (4096) -> per drafter layer `k_proj`, `k_norm`, RoPE at the token's position, `v_proj`: the drafter's KV cache, a
  2047-token window.
- **Block:** the bonus token at position p and mask tokens (id 154856) at p+1 .. p+7, embedded with GLM's own
  embedding (no scale). 5 Qwen3-style layers (hidden 4096, 32 query / 8 KV heads of 128, q/k RMSNorm, RoPE theta
  10000, MLP 12288, rms_norm_eps 1e-5), non-causal within the block, each sublayer wrapped by a two-tap grouped
  dynamic convolution over the block (groups of 16 channels, kernel from a projection of the sublayer's input).
- **Proposal:** the final norm's hidden for positions 1..7 -> GLM's LM head -> top 16 per position; a selector
  scores transitions between adjacent positions (rank-256 predecessor / successor codebooks, gated by a projection
  of the hidden) and the greedy walk gives 7 draft tokens (sampling: a walk with q for rejection sampling).
- **Verify:** the target on the 8 tokens [bonus, d1 .. d7]; the longest prefix matching its argmax is accepted,
  plus its next token as the new bonus; the accepted tokens' context features join the drafter's cache.

References (MIT): `z-lab/dflash` `dflash/model.py` (plain PyTorch, DFlash2DraftModel and a generate loop) and
SGLang's `python/sglang/srt/models/dflash.py` + `speculative/dflash_worker_v2.py`.

## What decides the gain

Measured verify costs (DECISIONS 2026-10-29, GLM, one sequence): a K-token step costs 1.39 / 1.74 / 2.16 / 2.97 /
3.59x a one-token step (31.7 ms at 6k) for K = 2 / 3 / 4 / 6 / 8: ~12.5 ms per extra token, the APU streaming its
cold experts. A draft step reads ~2.4 GB on a card (the drafter and GLM's head): ~4-5 ms. Speedup over plain decode
= accepted tokens per step / (step cost + draft cost). For 1.5x at K = 8 the drafter has to land ~5.7 of 8; at
K = 4, ~3.5 of 4. MTP (one draft, 83-86% accepted) gives ~1.3x on the same basis.

## Milestones

**F1: acceptance on our GLM (2-3 days, one GPU window of ~1 h).**
- `tools/ref/dflash2_ref.py`: the drafter and selector in plain PyTorch, ported from z-lab's model.py, with an
  offline greedy simulation: given a sequence's tokens and its target features, walk the generated part as
  speculative decoding would (draft 7, accept the matching prefix + bonus, advance) and report accepted tokens per
  step for verify widths 2-8, per position histograms.
- unholy: a feature dump in GLM's fast path (`UNHOLY_GLM_DFLASH_DUMP`: the mean over hc streams after the 5 layers,
  per token) and greedy continuations of ~30 public prompts (code, chat, math; our oracle prompts among them), run
  once on the box.
- Exit: accepted tokens per step at each verify width, the predicted speedup with the costs above. **Decision:**
  continue, or stop here.

**F2: the drafter on a card (2-3 days).** bf16 GEMMs / GEMVs (existing kernels), the block attention over the window
(GQA 32/8, 8 queries), the grouped convolutions, the head top-16 (GLM's EXL3 head GEMV + top-k), the selector.
Checked against `dflash2_ref.py` on dumped features.

**F3: GLM's multi-token verify (3-4 days).** Steps of 1 + 7 tokens per sequence (the decode GEMVs take up to 8 rows),
logits for every row, the features committed per accepted token, KDA's recurrent and conv states restored to the
accepted position (re-running the recurrence over the accepted tokens from the step's backup), DSA cache rows and
pool tail of rejected positions dropped. A forced full rejection must give logits bit-identical to plain decoding.

**F4: serving (2-3 days).** The scheduler's speculative path (DSV4's DSpark already verifies up to 6 tokens) with
DFlash2 for GLM: greedy, and rejection sampling with the selector's q. `--spec-drafts` picks the verify width.

**F5: measure and tune (1-2 days).** Served single-stream and concurrent tok/s against MTP and plain decode; verify
width by the selector's confidence; DECISIONS entry.

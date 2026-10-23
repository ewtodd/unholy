# Comparison protocol (draft, 2026-10-23)

How every candidate in `docs/PLAN-glm-dsv41.md` is measured, so the numbers line up across engines. Draft:
the items marked open are settled before the first comparison run.

## Subjects

| id | model | where | notes |
|---|---|---|---|
| `dsv4-local` | DeepSeek-V4-Flash-Vision-Exp, native | unholy, `:8200`, production flags | the reference row |
| `dsv4-hosted` | same model | OpenRouter, GMICloud or DeepInfra (fp8) | calibration: hosted vs local on identical weights |
| `glm-A` ... `glm-D` | GLM-5.3-Flash builds | unholy (after G6) | see `docs/glm53-flash-quant.md` |
| `glm-hosted` | GLM-5.3-Flash, full precision | OpenRouter, Z.AI (fp8) and Near AI (fp8) | two endpoints, logged per response |
| `v41-local` | DeepSeek-V4.1-Flash-1017, EXL3 3.0 | unholy (after D5) | SSD-streamed experts |
| `v41-hosted` | DeepSeek-V4.1-Flash-1017 | DeepSeek API, `deepseek-flash` | never the legacy model names |
| `qwen27b` | Qwen3.8-27B, fp8 | radiance, `:8100`, both R9700s | up to 352k context |
| `qwenfn` | Qwen3.8-Flash-Next, Q4 + MTP | gufo, `:8050`, Strix Halo | up to 262k context, 2 sessions |

## Intelligence

All through check-source, one item set per bench, identical for every subject.

| bench | items | seeds (local / hosted) | notes |
|---|---|---|---|
| Knoll | (owner's set) | 3 / - | the headline bench; local only until it is public |
| coding | ~150 recent LiveCodeBench problems | 2 / 4 | hidden tests in a sandbox without network (open: added with the check-source work) |
| AIME 24 / 25 / 26 | 90 | 2 / 5 | `\boxed{}` integer |
| MATH-500 | 200 | 2 / 4 | 4-shot Minerva |
| gsm8k-chat | 200 | 1 / 2 | near its ceiling (DSV4 flexible 0.98): a sanity check, not a ranking |
| longctx | 10 documents per depth: 32k, 128k, 512k (256k for the qwen models) | 1 / 1 | keyed-fact retrieval |

- **Reasoning:** every model at its "high" thinking level: `reasoning_effort: high` for DSV4 (unholy's
  template), GLM and V4.1 (plus `thinking: {type: enabled}` on the DeepSeek API), and the closest Qwen3.8
  template level (open: confirm which check-source level that is). One thinking-off run of gsm8k-chat per
  model for reference.
- **Sampling:** each model at its vendor's recommended settings, recorded in every result file; explicit
  per-request seeds. Truncated answers count as wrong, and the truncation rate is reported.
- **Statistics:** per-bench accuracy with a 95% binomial interval; model-vs-model differences paired on the
  same items and seeds (exact test on discordant pairs; paired bootstrap across seeds).
- **Hosted endpoints:** pinned, serving provider and returned model logged per item, off-endpoint items
  dropped; the logprob-returning endpoints are checked against our oracles once those exist.

## Performance

Measured through the OpenAI API so every engine is measured the same way, on an otherwise idle box (in the
qwen mode only the measured service gets traffic):

- **Prefill at depth:** time to first token over prompt length, `longctx --stream`, at 8k, 32k, 128k, 256k
  and 512k where the engine supports it, unique documents so prefix caches cannot help.
- **Decode, one stream:** tokens per second after the first token on streamed output, at ~4k context and at
  128k / 512k depth, median of 5, with each engine's production speculative-decoding setting (reported).
- **Concurrency:** gsm8k-chat at 1 / 2 / 4 / 8 concurrent requests: total completion tok/s and per-request
  decode tok/s.
- **Memory:** weights, KV and headroom as each engine reports them (`unholy plan`, the radiance and gufo
  startup logs).

Engines built here also get the internal measurements (`tools/box/curve.sh`, `bench.sh`, `eb.sh`).

## Outputs

Raw results under `/work/runs/<date>-<label>/` on the box; one report per comparison in `docs/` and a
DECISIONS entry with the numbers.

## Open

- check-source: API key, `--extra-body`, provider / model logging, the coding bench (deferred while the owner
  works on check-source).
- Knoll (owner).
- The per-model sampling table and the Qwen level mapping.

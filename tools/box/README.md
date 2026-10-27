# tools/box: dev scripts for son-of-anton

The scripts behind the numbers in STATUS / DECISIONS, kept in the repo so a cleanup of the box can't lose
them. Everything they write goes under `/work` (the SATA dev array); `/scratch` holds model weights only.

| path on the box | what |
|---|---|
| `/work/unholy-perf` | this checkout, synced and built by `sync.sh` (`.synced-rev` says which commit) |
| `/work/data` | inputs made once: `ids_code32768.bin` (+ `code-corpus.txt`), `gsm48.jsonl`, `oracle/code1024`, `base/`; GLM: `glm/` (code and chat ids, oracles `code1024` / `chat1024` and their `-sdpa` twins, `code4096`: 8 layers with top-k dumps) |
| `/work/runs/<date>-<label>` | one directory per run: logs, logits, outputs |
| `/work/env/py` | python with ROCm torch for the oracle (a GC root to a pinned env) |

Every script that needs the GPUs stops the llm-mode services first and restores the mode that was serving
when it exits, also on failure or Ctrl-C (`with_gpus` in `env.sh`). Paths and flags can be overridden through
the variables at the top of `env.sh`.

## Setup

```sh
tools/box/sync.sh                              # on the dev machine: rsync to /work/unholy-perf, build
ssh son-of-anton /work/unholy-perf/tools/box/setup.sh     # python env, token ids, request set
ssh son-of-anton /work/unholy-perf/tools/box/oracle.sh    # DSV4 oracle on the code prompt (GPU)
ssh son-of-anton /work/unholy-perf/tools/box/validate.sh --baseline
```

## Scripts

| script | does |
|---|---|
| `sync.sh [--no-build]` | (dev machine) copy the checkout to the box and build it |
| `setup.sh` | one-time inputs: `/work/env/py`, the code token ids (`mkids.py`), the GSM8K request set (`mkgsm.py`) |
| `mkchat.py` | a multi-turn GSM8K math chat through a model's chat template (`$GLM_PY`): the second oracle prompt |
| `oracle.sh [DIR]` | DeepSeek's model.py, layer-streamed on one R9700: 1024 prefill + 8 decode, logits + hidden states |
| `glm-oracle.sh [DIR] [flags]` | the same for GLM-5.3-Flash: transformers' glm5_next on the FP8 release, layer-streamed; also writes the routing counts; `--dump-topk` saves the DSA selections, `--prefill-chunk N` splits long prompts |
| `validate.sh [--baseline]` | resident-model logits vs the baseline (bit-identical expected) and vs the oracle; keep `MARGIN` fixed so the expert placement matches |
| `curve.sh LABEL` | `unholy-curve` to 512k: prefill by depth, decode probes |
| `bench.sh LABEL [flags]` | `unholy-bench` |
| `serve.sh [flags]` | the perf tree's server with the production flags on port 8300, foreground |
| `sweep.sh LABEL [flags]` | that server + check-source `gsm8k-chat` at 1 / 2 / 4 / 8 requests (check-source pinned in `env.sh`) |
| `eb.sh LABEL CONC [flags]` | `unholy-engine-batch` on `gsm48.jsonl` with logprobs; `REF=...jsonl` compares with an earlier run |
| `glm-ref.sh LABEL [flags]` | `unholy-glm-ref`: the GLM reference forward against `glm-oracle.sh`'s dumps, layer by layer, then KL / top-1 |
| `glm-serve.sh LABEL CMD...` | GLM build B served on port 8300 (production geometry), CMD run against it (`$URL`, `$RUN`), server stopped |
| `hold.sh CMD...` | run CMD (e.g. `bash -c 'a.sh && b.sh'`) with the services stopped once, not once per script |
| `trace-summary.py TRACE [flags]` | a rocprofv3 kernel trace (csv): per device, busy time and kernels by total time |
| `trace-grids.py TRACE [flags]` | the same trace by kernel and grid (blocks, VGPRs, LDS), flagging grids that fill a card's wave slots exactly |
| `isa.sh SRC ARCH OUT.s` | a source file's device assembly with the build's flags |

"""Token ids of chat prompts for DFlash2 acceptance runs on GLM-5.3-Flash: requests from the engine's request sets
(codegen48: code rewrites, gsm16_greedy: math reasoning, both thinking on), rendered by the checkpoint's own chat
template with the generation prompt, one raw int32 file per request.

    $GLM_PY tools/box/mkdflash-prompts.py --model /scratch/models/glm53-flash-exl3-4.05bpw \
        --out /work/data/glm/dflash-prompts /work/data/glm/codegen48.jsonl:16 /work/data/glm/gsm16_greedy.jsonl:16

FILE:N takes the first N requests of a set. Output: OUT/<set>-<i>.i32, the input of unholy-glm-features.
"""
import argparse
import json
import os

import numpy as np
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir with tokenizer.json and chat_template.jinja")
    ap.add_argument("--out", required=True)
    ap.add_argument("sets", nargs="+", help="request set .jsonl, optionally :N for its first N requests")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs(args.out, exist_ok=True)
    for spec in args.sets:
        path, _, count = spec.partition(":")
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            requests = [json.loads(line) for line in f if line.strip()]
        if count:
            requests = requests[: int(count)]
        for i, req in enumerate(requests):
            enc = tok.apply_chat_template(req["messages"], add_generation_prompt=True, tokenize=True,
                                          return_dict=True, **req.get("chat_template_kwargs", {}))
            ids = np.asarray(enc["input_ids"], dtype=np.int32)
            ids.tofile(os.path.join(args.out, "%s-%02d.i32" % (name, i)))
            print("%s-%02d: %d tokens" % (name, i, len(ids)))


if __name__ == "__main__":
    main()

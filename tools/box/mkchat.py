"""Token ids of a multi-turn math chat for the oracles: GSM8K test questions as user turns and their worked
answers as assistant turns, rendered by the model's own chat template, cut to --count tokens.

    $GLM_PY tools/box/mkchat.py --model /scratch/models/glm53-flash-fp8 --out /work/data/glm/ids_chat2048.bin

A second kind of text next to the code corpus (mkids.py), for KL against the oracle on prose, arithmetic and chat
markup. The answers are GSM8K's own, with the calculator annotations (<<48/2=24>>) removed. Rows come from the
Hugging Face datasets-server API; the template through transformers (the GLM venv has it).
"""
import argparse
import json
import re
import urllib.parse
import urllib.request

import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="checkpoint directory with the tokenizer and chat template")
ap.add_argument("--out", required=True)
ap.add_argument("--count", type=int, default=2048)
args = ap.parse_args()


def rows(split, offset, length):
    q = urllib.parse.urlencode({"dataset": "openai/gsm8k", "config": "main", "split": split,
                                "offset": offset, "length": length})
    with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + q, timeout=60) as r:
        return [x["row"] for x in json.load(r)["rows"]]


tok = AutoTokenizer.from_pretrained(args.model)
messages, ids = [], []
for r in rows("test", 0, 100):
    work, final = r["answer"].split("####")
    answer = re.sub(r"<<[^>]*>>", "", work).strip() + f"\n\nThe answer is {final.strip()}."
    messages += [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": answer}]
    ids = tok.apply_chat_template(messages, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if len(ids) >= args.count:
        break
if len(ids) < args.count:
    raise SystemExit(f"only {len(ids)} tokens from 100 questions")
np.asarray(ids[: args.count], dtype=np.uint32).tofile(args.out)
print(f"{args.out}: {args.count} tokens, {len(messages) // 2} turns; starts {tok.decode(ids[:24])!r}")

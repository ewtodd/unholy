"""Request set for unholy-engine-batch: GSM8K test questions as chat requests with a fixed 5-shot prompt.

    python3 tools/box/mkgsm.py --out /work/data/gsm48.jsonl [--count 48]
    python3 tools/box/mkgsm.py --out /work/data/glm/gsm-train96.jsonl --split train --offset 100 --count 96

The prompt is lm-eval's GSM8K format ("Question: ...\nAnswer: ...", the first five train items as shots) in
one user turn, thinking on, sampled with a per-request seed. Rows come from the Hugging Face datasets-server
API, so only the standard library is needed.
"""
import argparse
import json
import urllib.parse
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--count", type=int, default=48)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--split", default="test", help="questions from this split (train: offset past the shots)")
ap.add_argument("--offset", type=int, default=0)
args = ap.parse_args()


def rows(split, offset, length):
    q = urllib.parse.urlencode({"dataset": "openai/gsm8k", "config": "main", "split": split,
                                "offset": offset, "length": length})
    with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + q, timeout=60) as r:
        return [x["row"] for x in json.load(r)["rows"]]


shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in rows("train", 0, 5))
with open(args.out, "w") as f:
    for i, r in enumerate(rows(args.split, args.offset, args.count)):
        f.write(json.dumps({
            "messages": [{"role": "user", "content": shots + f"Question: {r['question']}\nAnswer:"}],
            "chat_template_kwargs": {"thinking": True},
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1000 + i,
            "max_tokens": args.max_tokens,
        }) + "\n")
print(f"wrote {args.count} requests to {args.out}")

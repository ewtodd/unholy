"""Request set where the model writes code (routing of code generation, next to mkgsm.py's reasoning): Python
standard library snippets (a file's first --lines lines, files in sorted path order from --skip), each with a request
to rewrite it with type hints and docstrings. Thinking on, sampled with a per-request seed.

    python3 tools/box/mkcodegen.py --out /work/data/glm/codegen48.jsonl [--count 48]
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--count", type=int, default=48)
ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--lines", type=int, default=60)
ap.add_argument("--max-tokens", type=int, default=1536)
args = ap.parse_args()

lib = sorted(glob.glob("/nix/store/*-python3-3.1[0-9].*/lib/python3.1[0-9]"))[0]
files = [p for p in sorted(glob.glob(os.path.join(lib, "*.py"))) if os.path.getsize(p) > 4000]
with open(args.out, "w") as f:
    for i, p in enumerate(files[args.skip:args.skip + args.count]):
        with open(p, encoding="utf-8", errors="replace") as src:
            code = "".join(src.readlines()[: args.lines])
        prompt = (f"Here is the start of Python's {os.path.basename(p)}:\n\n```python\n{code}```\n\n"
                  "Rewrite this code with full type hints and a docstring for every function and class. "
                  "Output the complete rewritten code.")
        f.write(json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"thinking": True},
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 2000 + i,
            "max_tokens": args.max_tokens,
        }) + "\n")
print(f"wrote {min(args.count, len(files) - args.skip)} requests to {args.out}")

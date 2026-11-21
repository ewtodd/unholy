"""Prefill curve of a running server: one request at a time on unique text (a random first line, then unholy's
sources, docs and tools from TREE, repeated to length), 64 tokens generated after each at temperature 0. The
server's own request lines (its log or journal) carry prompt tokens, time to first token, prefill and decode tok/s.

    /work/env/py tools/box/served-prefill.py [--url http://127.0.0.1:8270] [--tree /work/unholy-perf]
        [--targets 4096,32768,131072,524288]

A target is roughly the prompt's token count (3.3 characters a token on this text).
"""
import argparse
import glob
import json
import os
import time
import urllib.request
import uuid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8270")
    ap.add_argument("--tree", default="/work/unholy-perf")
    ap.add_argument("--targets", default="4096,32768,131072,524288")
    a = ap.parse_args()
    files = (sorted(glob.glob(f"{a.tree}/src/**/*.*", recursive=True))
             + sorted(glob.glob(f"{a.tree}/tools/**/*.*", recursive=True))
             + sorted(glob.glob(f"{a.tree}/docs/**/*.md", recursive=True)))
    corpus = "".join("// FILE " + f + "\n" + open(f, errors="replace").read() + "\n"
                     for f in files if os.path.getsize(f) < 2_000_000)
    print("corpus chars", len(corpus), flush=True)

    def text_of(chars):
        out, n = [], 0
        while n < chars:
            out.append(corpus[: chars - n])
            n += len(out[-1])
        return "".join(out)

    for target in (int(t) for t in a.targets.split(",")):
        prompt = (f"[run {uuid.uuid4()}]\n" + text_of(int(target * 3.3))
                  + "\n\nSummarize what the code above does in three sentences.")
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0}
        req = urllib.request.Request(f"{a.url}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.time()
        try:
            d = json.load(urllib.request.urlopen(req, timeout=3600))
            u = d["usage"]
            print(target, "->", u["prompt_tokens"], "prompt tokens,", u["prompt_tokens_details"]["cached_tokens"],
                  "cached,", round(time.time() - t, 1), "s", flush=True)
        except Exception as e:  # noqa: BLE001 - report and go on with the next length
            print(target, "error", e, flush=True)


if __name__ == "__main__":
    main()

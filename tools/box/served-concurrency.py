"""Aggregate decode throughput of a running server under concurrency: every request of an engine-batch request set
(jsonl, one chat request body per line; /work/data/gsm48.jsonl is the GSM8K set the engine-batch sweeps use), CONC at
a time, for each CONC in turn; completion tokens over wall time, as unholy-engine-batch reports it.

    /work/env/py tools/box/served-concurrency.py [--url http://127.0.0.1:8270] [--requests /work/data/gsm48.jsonl]
        [--concurrency 1,2,4,8] [--extra '{"reasoning_effort": "high"}']

--extra is merged into every request body.
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8270")
    ap.add_argument("--requests", default="/work/data/gsm48.jsonl")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--extra", default="{}")
    a = ap.parse_args()
    extra = json.loads(a.extra)
    bodies = [dict(json.loads(line), **extra) for line in open(a.requests) if line.strip()]

    def send(body):
        req = urllib.request.Request(f"{a.url}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        return d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"]

    for conc in (int(c) for c in a.concurrency.split(",")):
        t0 = time.time()
        with ThreadPoolExecutor(conc) as pool:
            res = list(pool.map(send, bodies))
        wall = time.time() - t0
        completion = sum(c for c, _ in res)
        print(f"{len(bodies)} requests, concurrency {conc}: {completion} completion tokens in {wall:.1f} s "
              f"({completion / wall:.1f} tok/s), {sum(p for _, p in res)} prompt tokens", flush=True)


if __name__ == "__main__":
    main()

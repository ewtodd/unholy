#!/usr/bin/env python3
"""Drive a mixed workload through an unholy server and save its routing histogram.

    routing_workload.py --url http://localhost:8100 --out profile.txt [--concurrency 4]
                        [--max-tokens 1200] [--files F ...] [--rounds 1]

The placement planner keeps each layer's most-used experts on the cards, so its profile should look
like the traffic the server will see. This sends chat requests across coding, math, English and
Chinese writing, knowledge questions, tool calling and long-document tasks (thinking on and off),
lets the model generate (decode routing dominates serving), then fetches /unholy/routing_profile.
The histogram is reset first.
"""
import argparse
import concurrent.futures as cf
import json
import random
import sys
import time
import urllib.request

PROMPTS = [
    # code
    "Write a C++20 lock-free single-producer single-consumer ring buffer with tests.",
    "Implement a Python async web crawler with a concurrency limit and robots.txt support.",
    "Explain Rust lifetimes with three increasingly tricky examples, then fix this: fn f<'a>(x: &'a str, y: &str) -> &'a str { y }",
    "Write a NixOS module for a systemd service that runs a Python web app with a dedicated user and a state directory.",
    "Review this code for bugs and suggest improvements:\n\nint* f(int n){int a[n];for(int i=0;i<=n;i++)a[i]=i;return a;}",
    "Write a HIP kernel that computes a row-wise softmax for a [rows, 4096] fp32 matrix, one block per row.",
    "Write a SQL schema and queries for a library system: books, members, loans, overdue reports.",
    "Implement Dijkstra's algorithm in Go with a binary heap and explain its complexity.",
    "Write a bash script that rotates logs older than 7 days, compresses them, and keeps at most 30 archives.",
    "Convert this JavaScript callback code to async/await and add error handling:\nfs.readFile('a', (e, d) => { if (e) throw e; fs.writeFile('b', d, () => {}); });",
    # math / reasoning
    "Prove that there are infinitely many primes of the form 4k+3.",
    "A bag has 5 red and 7 blue balls. Three are drawn without replacement. What is the probability that at least two are red? Show your work.",
    "Solve the differential equation y'' + 4y' + 4y = e^{-2x} with y(0)=1, y'(0)=0.",
    "How many ways can 8 non-attacking rooks be placed on a chessboard so that none is on the main diagonal?",
    "Explain the Fourier transform intuitively, then derive the transform of a Gaussian.",
    "If a train leaves at 3pm going 60 km/h and another at 4pm going 90 km/h on the same track, when does the second catch up? Then generalize.",
    # English writing / knowledge
    "Write a 600-word short story about a lighthouse keeper who finds a message in a bottle from the future.",
    "Summarize the causes and consequences of the 2008 financial crisis for a high-school student.",
    "Compare the leadership styles of Lincoln and Churchill during wartime.",
    "Draft a polite but firm email to a landlord about a broken heater that has not been fixed for two weeks.",
    "Explain how mRNA vaccines work and address three common misconceptions.",
    "What are the trade-offs between microservices and a monolith for a 10-person startup?",
    "Write a product description and FAQ for a smart water bottle.",
    "Explain the history of the Byzantine Empire in about 500 words.",
    # Chinese
    "请用中文详细解释量子纠缠的原理，并举一个通俗的例子。",
    "写一篇关于家乡秋天的散文，大约500字。",
    "请比较Python和Java在企业开发中的优缺点，用中文回答。",
    "帮我制定一个为期三个月的减脂健身计划，包括饮食建议。",
    "用中文解释什么是区块链，以及它在供应链管理中的应用。",
    "请分析《红楼梦》中林黛玉这个人物形象。",
    "解释一下Transformer模型中的注意力机制，并给出公式。",
    "写一个关于人工智能伦理的辩论稿，正方观点。",
    # other languages
    "Explique en français la différence entre l'imparfait et le passé composé avec des exemples.",
    "Erkläre auf Deutsch, wie eine Wärmepumpe funktioniert.",
    "日本語で、東京の観光名所を5つ紹介してください。",
    "Объясни по-русски, что такое рекурсия, с примером на Python.",
]

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                                                      "num_results": {"type": "integer"}},
                                                     "required": ["query"]}}},
    {"type": "function", "function": {"name": "run_python", "description": "Execute Python code and return stdout",
                                      "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                                     "required": ["code"]}}},
]
TOOL_PROMPTS = [
    "What's the weather in Berlin and Lisbon? Then suggest which city is better for a walk today.",
    "Find the latest stable Linux kernel version and summarize its main changes.",
    "Compute the 50th Fibonacci number and the sum of the first 1000 primes using Python.",
]


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    jobs = []
    for _ in range(args.rounds):
        for p in PROMPTS:
            jobs.append({"messages": [{"role": "user", "content": p}],
                         "chat_template_kwargs": {"thinking": rng.random() < 0.5}})
        for p in TOOL_PROMPTS:
            jobs.append({"messages": [{"role": "user", "content": p}], "tools": TOOLS})
        for path in args.files:
            text = open(path, encoding="utf-8", errors="replace").read()[:60000]
            task = rng.choice(["Summarize this document and list its key design decisions.",
                               "Find likely bugs in this code and explain them.",
                               "Write documentation for the main functions in this file."])
            jobs.append({"messages": [{"role": "user", "content": f"{task}\n\n```\n{text}\n```"}],
                         "chat_template_kwargs": {"thinking": rng.random() < 0.5}})
    rng.shuffle(jobs)
    reset = urllib.request.Request(args.url + "/unholy/routing_profile/reset", data=b"{}", method="POST")
    urllib.request.urlopen(reset).read()
    t0 = time.time()
    done = toks = 0

    def run(job):
        body = dict(job, max_tokens=args.max_tokens, temperature=0.7)
        r = post(args.url + "/v1/chat/completions", body)
        return r["usage"]

    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        for u in ex.map(run, jobs):
            done += 1
            toks += u["completion_tokens"]
            print(f"\r{done}/{len(jobs)} requests, {toks} generated tokens, {time.time() - t0:.0f} s", end="",
                  file=sys.stderr)
    print(file=sys.stderr)
    with urllib.request.urlopen(args.url + "/unholy/routing_profile") as r:
        prof = r.read().decode()
    open(args.out, "w").write(prof)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

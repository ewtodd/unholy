#!/usr/bin/env python3
"""Replay a recorded multi-turn conversation against an OpenAI-style chat server, as an agent would.

  replay-chat.py CONVERSATION [--url http://localhost:8300] [--max-tokens 256] [--max-context 32768]
                 [--model NAME] [--max-turns N] [--first-turn K] [--timeout S]

CONVERSATION is a JSON list of {"role", "content"} messages, or a Claude Code transcript (JSONL): its user and
assistant text turns, tool calls rendered as "[tool call NAME: {...}]" and tool results as "[tool result] ..." plain
text (consecutive messages of one role are merged). Every assistant message of the conversation is one turn: the
request carries the whole history before it (teacher-forced: the recording's own assistant turns, never the
model's), with max_tokens N, so the server prefills the new part and decodes N tokens (the routing trace records
both). --max-context stops before a turn whose history would exceed that many tokens (estimated at 3.2
characters per token, replaced by the server's own prompt_tokens count once known).
stdlib only; standard library JSON over urllib.
"""
import argparse
import json
import sys
import time
import urllib.request

CHARS_PER_TOKEN = 3.2


def _text_of(content):
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, str):
            out.append(b)
        elif not isinstance(b, dict):
            continue
        elif b.get("type") == "text":
            out.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            out.append(f"[tool call {b.get('name', '?')}: {json.dumps(b.get('input', {}), ensure_ascii=False)}]")
        elif b.get("type") == "tool_result":
            out.append("[tool result] " + _text_of(b.get("content")))
        # thinking, images: skipped
    return "\n".join(t for t in out if t)


def load(path):
    raw = open(path, encoding="utf-8").read()
    msgs = []
    if raw.lstrip().startswith("["):
        for m in json.loads(raw):
            msgs.append({"role": m["role"], "content": _text_of(m["content"])})
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            m = e.get("message")
            if e.get("type") not in ("user", "assistant") or not isinstance(m, dict) or e.get("isSidechain"):
                continue
            text = _text_of(m.get("content"))
            if text.strip():
                msgs.append({"role": m.get("role", e["type"]), "content": text})
    merged = []
    for m in msgs:
        if m["role"] not in ("user", "assistant", "system"):
            continue
        if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("conversation")
    ap.add_argument("--url", default="http://localhost:8300")
    ap.add_argument("--model", default="dsv4-flash")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-context", type=int, default=0, help="stop before a turn above this many tokens (0: no cap)")
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--first-turn", type=int, default=1, help="skip the replies before this assistant turn (still in the history)")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    msgs = load(a.conversation)
    turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0 and msgs[i - 1]["role"] != "assistant"]
    print(f"{a.conversation}: {len(msgs)} messages, {len(turns)} assistant turns", file=sys.stderr)
    done = 0
    t_all = time.time()
    ratio = CHARS_PER_TOKEN
    for n, i in enumerate(turns, 1):
        if n < a.first_turn:
            continue
        hist = msgs[:i]
        chars = sum(len(m["content"]) for m in hist)
        est = int(chars / ratio)
        if a.max_context and est + a.max_tokens > a.max_context:
            print(f"turn {n}: ~{est} tokens exceed --max-context {a.max_context}; stopping", file=sys.stderr)
            break
        t0 = time.time()
        r = post(a.url, {"model": a.model, "messages": hist, "max_tokens": a.max_tokens, "temperature": 0.0}, a.timeout)
        dt = time.time() - t0
        u = r.get("usage", {})
        pt = u.get("prompt_tokens", 0)
        if pt and chars:
            ratio = chars / pt
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", u.get("cached_tokens", 0))
        print(f"turn {n:3d}: prompt {pt:7d} (cached {cached}) completion {u.get('completion_tokens', 0):4d}  {dt:6.1f} s", flush=True)
        done += 1
        if a.max_turns and done >= a.max_turns:
            break
    print(f"{done} turns in {time.time() - t_all:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()

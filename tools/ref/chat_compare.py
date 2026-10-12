#!/usr/bin/env python3
"""Compare unholy's DSV4 chat encoding and output parsing with the Python references.

    chat_compare.py <unholy-text-test> <encoding_dir> [--random N] [--seed S]

encoding_dir holds the checkpoint's encoding_dsv4.py and tests/. Checks, in order: the golden
test_input_N/test_output_N pairs; randomized conversations (roles, tool calls, content blocks, tasks,
context splits, thinking/effort/drop flags) against encode_messages; parse_message_from_completion_text
on well-formed completions rendered by the reference itself, fed to our streaming parser in random
chunk sizes.
"""
import argparse
import copy
import json
import os
import random
import subprocess
import sys
import tempfile


def run(tool, cmd, items):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "in.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
        r = subprocess.run([tool, cmd, p], capture_output=True)
        if r.returncode:
            sys.exit(f"{cmd} failed: {r.stderr.decode()}")
        return json.loads(r.stdout)


TEXTS = ["hello", "", "What is 2+2?", "多行\n文本", "  spaces  ", "<think>literal</think>", "a\n\nb", "tab\there",
         "emoji 👍", "quote \" and \\ backslash", "｜DSML｜ inside", "<tool_result>x</tool_result>"]
VALUES = [1, -0.0, 0.1, 1e-07, 1e16, 12345678901234567890, 3.0, True, False, None, "str", "", "多",
          [1, 2, {"a": None}], {"nested": {"k": [1.5, "x"]}}, {"esc": "line\nbreak\t\"q\"\u0001"}]


def rand_args(rng):
    r = rng.random()
    if r < 0.1:
        return "not json {"
    if r < 0.15:
        return "[1, 2]"
    if r < 0.2:
        return {"dict": rng.choice(VALUES)}  # vLLM-style object arguments
    d = {}
    for _ in range(rng.randint(0, 4)):
        d[rng.choice(["location", "unit", "count", "q", "data", "参数", "x y"])] = copy.deepcopy(rng.choice(VALUES))
    return json.dumps(d, ensure_ascii=rng.random() < 0.5, indent=rng.choice([None, 2]))


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a place",
                                      "parameters": {"type": "object", "properties": {
                                          "location": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                                          "required": ["location"]}}},
    {"type": "function", "function": {"name": "search", "description": "搜索 the web; ratio 0.5 1e-05",
                                      "parameters": {"type": "object", "properties": {"q": {"type": "string"},
                                                                                      "count": {"type": "integer", "default": 10}}}}},
]


def rand_message(rng, role, call_ids):
    m = {"role": role}
    if role == "system":
        m["content"] = rng.choice(TEXTS + [None])
        if rng.random() < 0.2:
            m["tools"] = copy.deepcopy(TOOLS)
        if rng.random() < 0.1:
            m["response_format"] = {"type": "json_object", "schema": {"x": 1.0}}
    elif role == "developer":
        m["content"] = rng.choice(TEXTS[:4]) or "dev"
        if rng.random() < 0.3:
            m["tools"] = copy.deepcopy(TOOLS[:1])
    elif role == "user":
        r = rng.random()
        if r < 0.15:
            m["content"] = [{"type": "text", "text": rng.choice(TEXTS)} for _ in range(rng.randint(1, 3))]
        else:
            m["content"] = rng.choice(TEXTS)
        if rng.random() < 0.05:
            m["task"] = rng.choice(["action", "query", "title"])
        if rng.random() < 0.03:
            m["wo_eos"] = True
    elif role == "assistant":
        m["content"] = rng.choice(TEXTS + [None])
        if rng.random() < 0.6:
            m["reasoning_content"] = rng.choice(TEXTS)
        if rng.random() < 0.35:
            calls = []
            for k in range(rng.randint(1, 3)):
                cid = f"call_{len(call_ids)}"
                call_ids.append(cid)
                calls.append({"id": cid, "type": "function",
                              "function": {"name": rng.choice(["get_weather", "search"]), "arguments": rand_args(rng)}})
            m["tool_calls"] = calls
        if rng.random() < 0.05:
            m["wo_eos"] = True
    elif role == "tool":
        m["tool_call_id"] = rng.choice(call_ids) if call_ids and rng.random() < 0.9 else "unknown"
        m["content"] = rng.choice(TEXTS + [[{"type": "text", "text": "a"}, {"type": "other"}]])
    elif role == "latest_reminder":
        m["content"] = "2026-10-11,星期日"
    return m


def rand_conversation(rng):
    msgs, call_ids = [], []
    if rng.random() < 0.6:
        msgs.append(rand_message(rng, "system", call_ids))
    if rng.random() < 0.2:
        msgs.append(rand_message(rng, "latest_reminder", call_ids))
    for _ in range(rng.randint(1, 5)):
        msgs.append(rand_message(rng, rng.choice(["user"] * 5 + ["developer"]), call_ids))
        if rng.random() < 0.8:
            msgs.append(rand_message(rng, "assistant", call_ids))
            if msgs[-1].get("tool_calls"):
                calls = msgs[-1]["tool_calls"]
                for c in rng.sample(calls, len(calls)):  # results out of order: sorted by call order
                    msgs.append(rand_message(rng, "tool", [c["id"]]))
                if rng.random() < 0.5:
                    msgs.append(rand_message(rng, "assistant", call_ids))
    if rng.random() < 0.3 and msgs[-1]["role"] == "assistant":
        msgs.append(rand_message(rng, "user", call_ids))
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool")
    ap.add_argument("encoding_dir")
    ap.add_argument("--random", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    sys.path.insert(0, args.encoding_dir)
    import encoding_dsv4 as E

    rng = random.Random(args.seed)
    failures = 0

    # 1. Golden files.
    golden, expected = [], []
    tdir = os.path.join(args.encoding_dir, "tests")
    for n in range(1, 100):
        ip = os.path.join(tdir, f"test_input_{n}.json")
        if not os.path.exists(ip):
            break
        with open(os.path.join(tdir, f"test_output_{n}.txt"), encoding="utf-8") as f:
            want = f.read()
        for case in E.load_cases(ip):
            for mode in ("chat", "thinking"):
                if E.encode_messages(case["messages"], thinking_mode=mode) == want:
                    golden.append({"messages": case["messages"], "thinking_mode": mode})
                    expected.append(want)
                    break
            else:
                print(f"golden {n}: the reference itself does not reproduce test_output_{n}.txt")
                failures += 1
    got = run(args.tool, "chat", golden)
    for k, (g, w) in enumerate(zip(got, expected)):
        if g != w:
            failures += 1
            print(f"GOLDEN {k} differs:\n  ours {str(g)[:300]!r}\n  want {w[:300]!r}")
    print(f"golden: {len(golden) - failures}/{len(golden)} identical")

    # 2. Randomized conversations.
    cases, want = [], []
    skipped = 0
    for _ in range(args.random):
        msgs = rand_conversation(rng)
        opt = {"thinking_mode": rng.choice(["chat", "thinking"]),
               "reasoning_effort": rng.choice([None, "low", "high", "max"]),
               "drop_thinking": rng.random() < 0.8}
        ctx = None
        if rng.random() < 0.25 and len(msgs) > 2:
            cut = rng.randint(1, len(msgs) - 1)
            ctx, msgs = msgs[:cut], msgs[cut:]
        # Our encoder opens a turn after a trailing system message or one followed by an assistant
        # turn (vLLM's fix); the reference does not, so keep those out of the comparison.
        full = E.merge_tool_messages((ctx or []) + msgs)
        if opt["thinking_mode"] == "thinking" and opt["drop_thinking"] and not any(m.get("tools") for m in full):
            full = E._drop_thinking_messages(full)
        if any(m["role"] == "system" and (i + 1 == len(full) or full[i + 1]["role"] == "assistant")
               for i, m in enumerate(full)):
            skipped += 1
            continue
        # Object-valued tool-call arguments render as parameters (vLLM); the reference wraps them.
        if any(isinstance(c["function"]["arguments"], dict) for m in full for c in m.get("tool_calls") or []):
            skipped += 1
            continue
        try:
            w = E.encode_messages(copy.deepcopy(msgs), context=copy.deepcopy(ctx), **opt)
        except Exception as e:  # the reference rejects it; we only require that we do not crash
            w = {"error": repr(e)}
        cases.append({"messages": msgs, "context": ctx, **opt})
        want.append(w)
    got = run(args.tool, "chat", cases)
    bad = rejected = 0
    for c, g, w in zip(cases, got, want):
        if isinstance(w, dict):
            rejected += 1
            continue
        if g != w:
            bad += 1
            if bad <= 5:
                k = next((i for i in range(min(len(str(g)), len(w))) if str(g)[i] != w[i]), 0)
                print(f"RANDOM differs at {k}:\n  case {json.dumps(c, ensure_ascii=False)[:600]}\n"
                      f"  ours {str(g)[max(0, k - 80):k + 120]!r}\n  want {w[max(0, k - 80):k + 120]!r}")
    print(f"random: {len(cases) - rejected - bad}/{len(cases) - rejected} identical "
          f"({rejected} rejected by the reference, {skipped} cases with deliberate (vLLM) differences skipped)")
    failures += bad

    # 3. Completion parsing: reference-rendered assistant turns, parsed back.
    pcases, pwant = [], []
    for _ in range(args.random):
        thinking = rng.random() < 0.5
        msg = {"role": "assistant", "content": rng.choice(TEXTS[:5] + ["多行\n文本", "end\n"])}
        if thinking:
            msg["reasoning_content"] = rng.choice(TEXTS[:5] + ["step 1\nstep 2"])
        if rng.random() < 0.5:
            msg["tool_calls"] = [{"type": "function", "function": {
                "name": rng.choice(["get_weather", "search"]),
                "arguments": json.dumps({k: copy.deepcopy(rng.choice(VALUES)) for k in
                                         rng.sample(["location", "unit", "count", "q"], rng.randint(0, 3))},
                                        ensure_ascii=False)}} for _ in range(rng.randint(1, 3))]
        text = E.render_message(1, [{"role": "user", "content": "q"}, msg],
                                thinking_mode="thinking" if thinking else "chat")
        try:
            ref = E.parse_message_from_completion_text(text, "thinking" if thinking else "chat")
        except Exception:
            continue
        pcases.append({"text": text, "thinking": thinking, "chunk": rng.choice([1, 2, 3, 5, 8, 13, 1 << 20])})
        pwant.append(ref)
    got = run(args.tool, "parse", pcases)
    bad = 0
    for c, g, w in zip(pcases, got, pwant):
        wc = [{"name": t["function"]["name"], "arguments": json.dumps(json.loads(t["function"]["arguments"]),
                                                                       ensure_ascii=False)}
              for t in w["tool_calls"]]
        if g["reasoning"] != w["reasoning_content"] or g["content"] != w["content"] or g["tool_calls"] != wc:
            bad += 1
            if bad <= 5:
                print(f"PARSE differs:\n  text {c['text']!r} chunk {c['chunk']}\n  ours {g}\n  want {w}")
    print(f"parse: {len(pcases) - bad}/{len(pcases)} identical")
    failures += bad
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

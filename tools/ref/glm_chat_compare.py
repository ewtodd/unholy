"""Compare unholy's GLM-5.3-Flash chat format with the references: transformers' apply_chat_template on the
checkpoint's chat_template.jinja (rendering) and vLLM's GLM-4.7 parser (completion parsing).

    /work/env/glm/bin/python tools/ref/glm_chat_compare.py <unholy-text-test> <glm checkpoint dir> [--random N] [--seed S]

Rendering: golden conversations plus random ones over every message shape the template handles. The reference
gets tool-call arguments as objects (vLLM converts JSON text before rendering; unholy does that itself, so half
the random cases send it text). Parsing: random completions through vLLM's Glm47MoeParser.parse and through
unholy's parser fed in random chunk sizes. Cases on the documented deviations (calls to undeclared tools, calls
cut off by the length limit) are counted separately, not compared.
"""
import argparse
import copy
import json
import os
import random
import subprocess
import sys
import tempfile

import transformers
from transformers import AutoTokenizer

transformers.logging.set_verbosity_error()


def run_unholy(binary, cmd, items):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(items, f, ensure_ascii=False)
        path = f.name
    try:
        out = subprocess.run([binary, cmd, path], capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    return json.loads(out)


# ---- random conversations ---------------------------------------------------------------------------

WORDS = ["the", "weather", "Oslo", "código", "数据", "naïve", "  ", "\n", "\t", "<think>", "</think>", "x", "42",
         "{", "}", "\"q\"", "tool", "émoji 🚀", "end.", "  ", "<tool_call>", "a\nb"]


def text(r, n=6):
    return "".join(r.choice(WORDS) + (" " if r.random() < 0.5 else "") for _ in range(r.randint(0, n)))


def value(r, depth=0):
    k = r.randint(0, 6 if depth < 2 else 3)
    if k == 0:
        return text(r, 3)
    if k == 1:
        return r.randint(-10**6, 10**6)
    if k == 2:
        return r.choice([0.5, -2.25, 1e-7, 3.0, 12345.678])
    if k == 3:
        return r.choice([True, False, None])
    if k == 4:
        return [value(r, depth + 1) for _ in range(r.randint(0, 3))]
    return {r.choice(["a", "b", "ü", "k k"]): value(r, depth + 1) for _ in range(r.randint(0, 3))}


def content(r):
    k = r.randint(0, 5)
    if k == 0:
        return None
    if k <= 2:
        return text(r)
    parts = []
    for _ in range(r.randint(0, 4)):
        t = r.randint(0, 5)
        if t <= 1:
            parts.append({"type": "text", "text": text(r)})
        elif t == 2:
            parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})
        elif t == 3:
            parts.append(text(r))
        elif t == 4:
            parts.append({"type": "video", "video": "v.mp4"})
        else:
            parts.append({"type": "input_audio", "input_audio": {}})
    return parts


def tools_list(r):
    out = []
    for i in range(r.randint(0, 3)):
        fn = {"name": f"fn_{i}", "description": text(r, 3),
              "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "n": {"type": "integer"}}}}
        if r.random() < 0.2:
            fn["strict"] = True
        if r.random() < 0.2:
            fn["defer_loading"] = r.random() < 0.5
        out.append({"type": "function", "function": fn} if r.random() < 0.8 else fn)
    return out


def conversation(r, tools):
    msgs, ids = [], 0
    for _ in range(r.randint(1, 7)):
        role = r.choice(["user", "user", "assistant", "assistant", "tool", "system"])
        if role == "assistant":
            m = {"role": "assistant", "content": content(r)}
            if r.random() < 0.3:
                m["content"] = text(r) + "<think>" + text(r) + "</think>" + text(r)
            if r.random() < 0.4:
                m["reasoning_content"] = text(r)
            if r.random() < 0.5:
                calls = []
                for _ in range(r.randint(1, 3)):
                    ids += 1
                    args = {r.choice(["a", "n", "b", "città"]): value(r) for _ in range(r.randint(0, 3))}
                    call = {"type": "function", "function": {"name": f"fn_{r.randint(0, 2)}", "arguments": args}}
                    if r.random() < 0.85:
                        call["id"] = f"call_{ids}"
                    calls.append(call)
                m["tool_calls"] = calls
            msgs.append(m)
        elif role == "tool":
            prev = [c.get("id", "") for c in (msgs[-1].get("tool_calls", []) if msgs and msgs[-1]["role"] == "assistant" else [])]
            for _ in range(r.randint(1, 3)):
                m = {"role": "tool"}
                k = r.randint(0, 9)
                if k <= 4:
                    m["content"] = text(r)
                elif k <= 6:
                    m["content"] = content(r)
                elif k == 7:
                    m["content"] = [{"output": text(r), "tool_call_id": r.choice(prev or ["zz"])} for _ in range(r.randint(1, 2))]
                elif k == 8 and tools:
                    m["content"] = [{"type": "tool_reference", "name": f"fn_{r.randint(0, 3)}"}]
                else:
                    m["content"] = text(r)
                if r.random() < 0.8:
                    m["tool_call_id"] = r.choice(prev + ["call_x"]) if prev else "call_y"
                msgs.append(m)
        else:
            msgs.append({"role": role, "content": content(r)})
    if r.random() < 0.3:
        r.shuffle(msgs)
    return msgs


def golden():
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Oslo?"},
            {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\", \"days\": 2}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 3}"},
            {"role": "assistant", "content": "<think>sum it</think>It is 3 C."},
            {"role": "user", "content": [{"type": "text", "text": "And "}, {"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "this?"}]}]
    return [dict(messages=msgs, tools=tools), dict(messages=msgs, tools=tools, reasoning_effort="high"),
            dict(messages=msgs, tools=tools, reasoning_effort="low", chat_template_kwargs={"clear_thinking": True}),
            dict(messages=[{"role": "user", "content": "hi"}])]


def reference_render(tok, req):
    msgs = copy.deepcopy(req["messages"])
    for m in msgs:  # vLLM's conversion of tool-call arguments given as text
        for c in m.get("tool_calls") or []:
            if isinstance(c.get("function", {}).get("arguments"), str):
                c["function"]["arguments"] = json.loads(c["function"]["arguments"])
    kw = dict(req.get("chat_template_kwargs") or {})
    if req.get("reasoning_effort") is not None:
        kw["reasoning_effort"] = req["reasoning_effort"]
    return tok.apply_chat_template(msgs, tools=req.get("tools") or None, add_generation_prompt=True, tokenize=False, **kw)


def compare_render(binary, tok, reqs, label):
    expected = []
    for req in reqs:
        try:
            expected.append(reference_render(tok, req))
        except Exception as e:  # noqa: BLE001  the template rejects it
            expected.append({"error": str(e)})
    got = run_unholy(binary, "glm-request", reqs)
    same = rejected = 0
    for req, e, g in zip(reqs, expected, got):
        if isinstance(e, dict):
            rejected += 1
            continue
        if e == g:
            same += 1
        elif same + rejected < 10 ** 9:
            print(f"MISMATCH {label}:\n  request  {json.dumps(req, ensure_ascii=False)[:600]}\n  expected {e!r}\n  got      {g!r}")
    print(f"{label}: {same}/{len(reqs) - rejected} identical ({rejected} rejected by the template)")
    return same == len(reqs) - rejected


# ---- random completions ------------------------------------------------------------------------------

def completion(r, tools):
    parts = [text(r)]
    if r.random() < 0.85:
        parts.append("</think>")
        parts.append(text(r))
    for _ in range(r.choice([0, 0, 1, 1, 2, 3])):
        names = [t.get("function", t)["name"] for t in tools] or ["fn_0"]
        call = "<tool_call>" + r.choice(["", " ", "\n"]) + r.choice(names) + r.choice(["", "\n"])
        for _ in range(r.randint(0, 3)):
            k = r.choice(["a", "n", " b ", "a"])
            v = r.choice([text(r, 2), str(r.randint(-99, 99)), "3.0", "1_000", "null", "true", "[1, 2]", '{"x": 1.50}', " 7 "])
            call += f"<arg_key>{k}</arg_key>" + r.choice(["", "\n", " "]) + f"<arg_value>{v}</arg_value>" + r.choice(["", "\n"])
        call += "</tool_call>"
        parts.append(call + r.choice(["", "\n", text(r, 2)]))
    return "".join(parts)


def compare_parse(binary, tok, n, r):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.glm47_moe import Glm47MoeParser

    cases, refs, skipped = [], [], 0
    for _ in range(n):
        tools = [t if "function" in t else {"type": "function", "function": t}  # vLLM's request model wants these
                 for t in (tools_list(r) if r.random() < 0.7 else [])]
        for t in tools:
            fn = t.get("function", t)
            fn["parameters"]["properties"].update({"b": {"type": ["boolean", "null"]}, "f": {"type": "number"}})
            fn.pop("defer_loading", None)
        text_ = completion(r, tools)
        req = ChatCompletionRequest(model="glm", messages=[{"role": "user", "content": "x"}], tools=tools or None)
        reasoning, content_, calls = Glm47MoeParser(tok, tools=req.tools).parse(text_, req, enable_auto_tools=True)
        declared = {t.get("function", t)["name"] for t in tools}
        if tools and any(seg.split("<arg_key>")[0].split("</tool_call>")[0].strip() not in declared
                         for seg in text_.split("<tool_call>")[1:]):
            skipped += 1  # documented deviation: calls to undeclared tools
            continue
        cases.append({"text": text_, "chunk": r.choice([1, 2, 3, 7, 64, 1 << 20]), "tools": tools})
        refs.append({"reasoning": reasoning or "", "content": content_ or "",
                     "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in (calls or [])]})
    got = run_unholy(binary, "glm-parse", cases)
    same = lead = 0
    for c, e, g in zip(cases, refs, got):
        if e == g:
            same += 1
        elif (e["tool_calls"] and g["tool_calls"] == e["tool_calls"] and g["reasoning"] == e["reasoning"]
              and g["content"].lstrip() == e["content"]):
            lead += 1  # documented: a stream has sent the content's leading whitespace before the call shows up
        else:
            print(f"MISMATCH parse:\n  text     {c['text']!r}\n  expected {e}\n  got      {g}")
    print(f"parse: {same + lead}/{len(cases)} identical ({lead} of them up to the content's leading whitespace "
          f"before a call; {skipped} with calls to undeclared tools not compared)")
    return same + lead == len(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("ckpt")
    ap.add_argument("--random", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    r = random.Random(args.seed)
    ok = compare_render(args.binary, tok, golden(), "golden")
    reqs = []
    for _ in range(args.random):
        tools = tools_list(r)
        req = {"messages": conversation(r, tools)}
        if tools:
            req["tools"] = tools
        if r.random() < 0.5:
            for m in req["messages"]:
                for c in m.get("tool_calls") or []:
                    c["function"]["arguments"] = json.dumps(c["function"]["arguments"], ensure_ascii=False)
        e = r.choice([None, "low", "high", "max", "medium"])
        if e is not None:
            req["reasoning_effort"] = e
        if r.random() < 0.3:
            req["chat_template_kwargs"] = {"clear_thinking": r.random() < 0.7}
        reqs.append(req)
    ok &= compare_render(args.binary, tok, reqs, "random")
    ok &= compare_parse(args.binary, tok, args.random, r)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

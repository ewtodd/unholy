"""Compare unholy's Hy4-flash chat format with the references: transformers' apply_chat_template on the
checkpoint's chat_template.jinja (rendering) and vLLM's hy_v4 reasoning and tool parsers (completion parsing).

    env/bin/python tools/ref/hy4_chat_compare.py <unholy-text-test> <hy4 checkpoint dir> [--random N] [--seed S]
                                                  [--parse N]

Pinned references (the box's /work/hy4-text/env):
  transformers  huggingface/transformers PR #50903, head ManaEstras/transformers@b865850d05326cf20e306b31f14b0027de802da3
  checkpoint    tencent/Hy4-flash-FP8 @ bd0e136e89a9ecd5ea17668cd89cf54fa127e795 (chat_template.jinja, tokenizer.json)
  vLLM          0.29.0, vllm/reasoning/hy_v4_reasoning_parser.py (HYV4ReasoningExtractor) and
                vllm/tool_parsers/hy_v4_tool_parser.py (HYV4ToolExtractor, strict=True as HYV4ToolParser builds it),
                combined as vllm/parser/abstract_parser.py DelegatingParser.parse does: reasoning first, then the
                tool parser on the content, whitespace-only content dropped when calls were found.

Rendering: golden conversations plus random ones over every message shape the template handles (tools and tool
calls, tool results as text / blocks / tool_search_output, ids that do and do not allow the re-ordering, system
prompts, images and other blocks, reasoning / reasoning_content, both efforts, the template's other kwargs). The
reference gets tool-call arguments as objects (vLLM converts JSON text before rendering; unholy does that
itself, so half the random cases send it text). A request the template rejects must be rejected by unholy too.
Parsing: random completions through vLLM's extractors and through unholy's parser fed in random chunk sizes
(thinking and no_think). Cases on the documented departures are counted separately, not compared: a tool block
cut off by the length limit (unholy salvages the calls), and a call whose name is neither declared nor
[A-Za-z0-9_-]{1,64} (unholy returns the block as content, vLLM emits the call). tools/text_test/hy4_parse_test.cpp
covers those, without vLLM. Not generated: <think> anywhere but the very start of a completion (vLLM drops the
text before it, a stream cannot), and non-ASCII digits in argument values (Python's int() reads them).
"""
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
import argparse
import copy
import json
import random
import re
import subprocess
import sys
import tempfile

import transformers
from transformers import AutoTokenizer

transformers.logging.set_verbosity_error()

SUF = ":opensource"


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
         "{", "}", "\"q\"", "tool", "émoji 🚀", "end.", "  ", "<tool_calls>", "a\nb", f"<think{SUF}>",
         f"</think{SUF}>", f"<tool_call{SUF}>", f"<arg_key{SUF}>", "'", "\\", "ü", " ", " "]


def text(r, n=6):
    return "".join(r.choice(WORDS) + (" " if r.random() < 0.5 else "") for _ in range(r.randint(0, n)))


def value(r, depth=0):
    k = r.randint(0, 6 if depth < 2 else 3)
    if k == 0:
        return text(r, 3)
    if k == 1:
        return r.randint(-10**6, 10**6)
    if k == 2:
        return r.choice([0.5, -2.25, 1e-7, 3.0, 12345.678, 1e22])
    if k == 3:
        return r.choice([True, False, None])
    if k == 4:
        return [value(r, depth + 1) for _ in range(r.randint(0, 3))]
    return {r.choice(["a", "b", "ü", "k k"]): value(r, depth + 1) for _ in range(r.randint(0, 3))}


IMG = [0]


def image_block(r):
    IMG[0] += 1
    u = f"http://x/{IMG[0]}.png"
    k = r.randint(0, 3)
    if k == 0:
        return {"type": "image_url", "image_url": {"url": u}}
    if k == 1:
        return {"type": "image_url", "image_url": u}
    if k == 2:
        return {"type": "image", "image": u}
    return {"image_url": {"url": u}}  # no type: the template looks at the key


def content(r):
    k = r.randint(0, 6)
    if k == 0:
        return None
    if k <= 2:
        return text(r)
    if k == 3 and r.random() < 0.3:
        return r.choice([5, 1.5, True, {"type": "text", "text": "x"}, {"a": [1, "b", None]}])
    parts = []
    for _ in range(r.randint(0, 4)):
        t = r.randint(0, 8)
        if t <= 1:
            parts.append({"type": "text", "text": text(r)})
        elif t == 2:
            parts.append(image_block(r))
        elif t == 3:
            parts.append(text(r))
        elif t == 4:
            parts.append({"type": r.choice(["video", "video_url"]), "video": "v.mp4"})
        elif t == 5:
            parts.append({"type": "input_audio", "input_audio": {"data": "ab'c", "format": "wav"}})
        elif t == 6:
            parts.append(r.choice([1, 2.5, None, True, ["a", 1]]))
        elif t == 7:
            parts.append({"type": "text"})
        else:
            parts.append({"video_url": {"url": "v"}})
    return parts


def tools_list(r):
    out = []
    for i in range(r.randint(0, 3)):
        fn = {"name": f"fn_{i}", "description": text(r, 3),
              "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "n": {"type": "integer"}}}}
        if r.random() < 0.2:
            fn["strict"] = True
        if r.random() < 0.25:
            fn["defer_loading"] = r.random() < 0.5
        out.append({"type": "function", "function": fn} if r.random() < 0.8 else fn)
    return out


def tool_content(r, prev, tools):
    k = r.randint(0, 11)
    if k <= 4:
        return text(r)
    if k <= 5:
        return content(r)
    if k == 6:
        return [{"output": text(r), "tool_call_id": r.choice(prev or ["zz"])} for _ in range(r.randint(1, 2))]
    if k in (7, 8):
        items = []
        for _ in range(r.randint(1, 4)):
            t = r.randint(0, 4)
            if t <= 1:
                items.append({"type": "tool_search_output", "tools": tools_list(r)})
            elif t == 2:
                items.append({"type": "text", "text": text(r)})
            elif t == 3:
                items.append({"output": content(r)})
            else:
                items.append(r.choice([text(r), image_block(r), {"type": "tool_search_output"}]))
        return items
    if k == 9:
        return {"result": text(r, 2)}
    return text(r)


def conversation(r, tools):
    msgs, ids = [], 0
    for _ in range(r.randint(1, 7)):
        role = r.choice(["user", "user", "assistant", "assistant", "tool", "system", "developer"])
        if role == "assistant":
            m = {"role": "assistant", "content": content(r)}
            if r.random() < 0.3:
                m["content"] = text(r) + "<think>" + text(r) + "</think>" + text(r)
            q = r.random()
            if q < 0.3:
                m["reasoning_content"] = text(r)
            elif q < 0.5:
                m["reasoning"] = text(r)
            elif q < 0.6:
                m["reasoning"] = ""
                m["reasoning_content"] = text(r)
            elif q < 0.65:
                m["reasoning"] = None
            if r.random() < 0.5:
                calls = []
                for _ in range(r.randint(1, 3)):
                    ids += 1
                    args = {r.choice(["a", "n", "b", "città"]): value(r) for _ in range(r.randint(0, 3))}
                    call = {"type": "function", "function": {"name": f"fn_{r.randint(0, 2)}", "arguments": args}}
                    z = r.random()
                    if z < 0.75:
                        call["id"] = f"call_{ids}"
                    elif z < 0.85:
                        call["tool_call_id"] = f"call_{ids}"
                    elif z < 0.9:
                        call["id"] = ""
                    if r.random() < 0.1:
                        call = call["function"]
                    calls.append(call)
                if r.random() < 0.1 and len(calls) > 1:
                    calls[1]["id"] = calls[0].get("id", "dup")  # duplicate ids
                m["tool_calls"] = calls
            elif r.random() < 0.05:
                m["tool_calls"] = []
            msgs.append(m)
        elif role == "tool":
            prev = []
            if msgs and msgs[-1]["role"] == "assistant":
                for c in msgs[-1].get("tool_calls", []):
                    i = c.get("id") or c.get("tool_call_id")
                    if i:
                        prev.append(i)
            for _ in range(r.randint(1, 3)):
                m = {"role": "tool", "content": tool_content(r, prev, tools)}
                z = r.random()
                if z < 0.6:
                    m["tool_call_id"] = r.choice(prev + ["call_x"]) if prev else "call_y"
                elif z < 0.75:
                    m["id"] = r.choice(prev + ["call_x"]) if prev else "call_y"
                elif z < 0.8:
                    m["tool_call_id"] = None
                if r.random() < 0.05:
                    del m["content"]
                msgs.append(m)
        else:
            msgs.append({"role": role, "content": content(r)})
    if r.random() < 0.3:
        r.shuffle(msgs)
    # a faithful continuation of the last assistant turn's calls in the right order, most of the time
    if r.random() < 0.5:
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and m.get("tool_calls") and r.random() < 0.7:
                good = [{"role": "tool", "tool_call_id": c.get("id") or c.get("tool_call_id"), "content": text(r)}
                        for c in m["tool_calls"] if (c.get("id") or c.get("tool_call_id"))]
                r.shuffle(good)
                msgs[i + 1:i + 1] = good
                break
    return msgs


def golden():
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Oslo?"},
            {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\", \"days\": 2}"}},
                            {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "{\"temp\": 9}"},
            {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 3}"},
            {"role": "assistant", "content": "<think>sum it</think>It is 3 C.", "reasoning": "It is 3 and 9."},
            {"role": "user", "content": [{"type": "text", "text": "And "}, {"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "this?"}]}]
    plain = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello", "reasoning_content": "greet"},
             {"role": "user", "content": "bye"}]
    return [dict(messages=msgs, tools=tools), dict(messages=msgs, tools=tools, reasoning_effort="no_think"),
            dict(messages=msgs), dict(messages=msgs, reasoning_effort="no_think"),
            dict(messages=plain), dict(messages=plain, reasoning_effort="no_think"),
            dict(messages=plain, chat_template_kwargs={"reasoning_effort": "no_think", "preserved_thinking": True}),
            dict(messages=plain, tools=tools, chat_template_kwargs={"preserved_thinking": False}),
            dict(messages=plain + [{"role": "assistant", "content": "partial"}],
                 chat_template_kwargs={"raw_last_assistant": True}),
            dict(messages=msgs[:3], tools=tools, chat_template_kwargs={"fallback_strategy": "reasoning_toolcall_retry"}),
            dict(messages=[dict(role="user", content=[{"type": "image_url", "image_url": "u"}] * 2)],
                 chat_template_kwargs={"add_vision_id": True}),
            dict(messages=[{"role": "system", "content": "sys"}] + plain, tools=tools),
            dict(messages=[]), dict(messages=[{"role": "user", "content": "hi"}], reasoning_effort="low"),
            dict(messages=[{"role": "user", "content": "hi"}], reasoning_effort="auto")]


def reference_render(tok, req):
    msgs = copy.deepcopy(req["messages"])
    for m in msgs:  # vLLM's conversion of tool-call arguments given as text
        for c in m.get("tool_calls") or []:
            if isinstance(c.get("function", {}).get("arguments"), str):
                c["function"]["arguments"] = json.loads(c["function"]["arguments"])
    kw = dict(req.get("chat_template_kwargs") or {})
    e = req.get("reasoning_effort")
    if e is not None and e != "auto":  # vLLM's merge_kwargs drops None and "auto"
        kw["reasoning_effort"] = e
    return tok.apply_chat_template(msgs, tools=req.get("tools") or None, add_generation_prompt=True, tokenize=False, **kw)


def compare_render(binary, tok, reqs, label):
    expected = []
    for req in reqs:
        try:
            expected.append(reference_render(tok, req))
        except Exception as e:  # noqa: BLE001  the template rejects it
            expected.append({"error": str(e)})
    got = run_unholy(binary, "hy4-request", reqs)
    same = rejected = bad = 0
    for req, e, g in zip(reqs, expected, got):
        if isinstance(e, dict):
            if "error" in g:
                rejected += 1
            else:
                bad += 1
                print(f"MISMATCH {label}: the template rejects this ({e['error'][:100]}) but unholy renders it:\n"
                      f"  request {json.dumps(req, ensure_ascii=False)[:600]}")
            continue
        if "error" in g:
            bad += 1
            print(f"MISMATCH {label}: unholy rejects ({g['error']}):\n  request {json.dumps(req, ensure_ascii=False)[:600]}")
            continue
        n_img = e.count("hy_image▁pad")
        msgs = req["messages"]
        # the generation prompt opens <think>; not after a last assistant message, nor on a fallback retry
        think = (e.endswith(f"<think{SUF}>") and msgs[-1].get("role") != "assistant"
                 and (req.get("chat_template_kwargs") or {}).get("fallback_strategy") != "reasoning_toolcall_retry")
        if e == g["prompt"] and len(g["images"]) == n_img and g["thinking"] == think:
            same += 1
        else:
            bad += 1
            if bad <= 8:
                print(f"MISMATCH {label}:\n  request  {json.dumps(req, ensure_ascii=False)[:600]}\n  expected {e!r}\n"
                      f"  got      {g['prompt']!r}\n  images {g['images']} (expect {n_img}), thinking {g['thinking']} (expect {think})")
    print(f"{label}: {same}/{len(reqs) - rejected} identical ({rejected} rejected by both)"
          + ("" if not bad else f", {bad} MISMATCHES"))
    return bad == 0


# ---- random completions ------------------------------------------------------------------------------

CWORDS = [w for w in WORDS if w not in ("<think>", f"<think{SUF}>")]


def ctext(r, n=6):
    return "".join(r.choice(CWORDS) + (" " if r.random() < 0.5 else "") for _ in range(r.randint(0, n)))

TOOLS = [
    {"name": "fn_0", "parameters": {"type": "object", "properties": {
        "a": {"type": "string"}, "n": {"type": "integer"}, "b": {"type": ["boolean", "null"]}, "f": {"type": "number"},
        "o": {"type": "object"}, "l": {"type": "array"}, "u": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        "p": {"type": "int"}, "q": {"type": "Integer"}, "x": {}, "e": {"enum": ["a", "b"]},
        "m": {"oneOf": [{"type": "boolean"}, {"type": "array"}]}, "z": {"type": "null"}, "k": {"type": "unsigned32"}}}},
    {"name": "fn_1", "parameters": None},
    {"name": "fn_2"},
]
VALUES = ["", "x", "hello world", " 7 ", "7", "-3", "+5", "007", "-0", "1_000", "3.0", "1e3", "2.5e-3", "1e999", ".5",
          "5.", "true", "True", "FALSE", "null", "None", "[1, 2]", '{"x": 1.50}', "{'a': 1}", "(1, 2)", "(1)", "()",
          "'x'", '"x y"', "'a\\nb'", "0x10", "NaN", "Infinity", "-Infinity", "[NaN, 1]", "1j", "{1, 2}", "b'x'",
          "[1, 2", "é", "tab\there", "a</arg_key>b", "<tool_call>", "x<arg_value>y", "  \n", "12345678901234567890123",
          "1.0e+22", "{}", "[]", '["a", {"b": null}]', "yes", "3 4"]


def arg(r):
    k = r.choice(["a", "n", " b ", "f", "o", "l", "u", "p", "q", "x", "e", "m", "z", "k", "a", "c"])
    v = r.choice(VALUES + [ctext(r, 2)])
    return k, v


def call(r, names, bad_name):
    name = r.choice(names)
    if bad_name:
        name = r.choice(["get weather", "a.b", "", "x" * 70, "é"])
    s = f"<tool_call{SUF}>" + r.choice(["", " ", "\n"]) + name + r.choice(["", "\n", " "])
    for _ in range(r.randint(0, 3)):
        k, v = arg(r)
        s += f"<arg_key{SUF}>{k}</arg_key{SUF}>" + r.choice(["", "", "", "\n"]) + f"<arg_value{SUF}>{v}</arg_value{SUF}>" + r.choice(["", "", "\n"])
    return s + f"</tool_call{SUF}>"


def completion(r, tools, thinking):
    names = [t["name"] for t in tools] or ["fn_0"]
    parts = []
    if thinking:
        if r.random() < 0.1:
            parts.append(f"<think{SUF}>")
        parts.append("".join(r.choice(CWORDS) + r.choice(["", " "]) for _ in range(r.randint(0, 5))))
        if r.random() < 0.9:
            parts.append(f"</think{SUF}>")
        else:
            return "".join(parts)
    parts.append(r.choice(["", "", "\n", " ", ctext(r, 4)]))
    if r.random() < 0.05:
        parts.append(f"</tool_calls{SUF}>")
    for _ in range(r.choice([0, 1, 1, 1, 2])):
        block = f"<tool_calls{SUF}>" + r.choice(["", "", "\n"])
        for _ in range(r.randint(0, 3) if r.random() < 0.1 else r.randint(1, 3)):
            block += call(r, names, r.random() < 0.03) + r.choice(["", "", "\n", "junk"])
        if r.random() < 0.08:  # corrupt it
            m = r.randint(0, 3)
            if m == 0:
                block = block.replace(f"</tool_call{SUF}>", "", 1)
            elif m == 1:
                block = block.replace(f"</arg_value{SUF}>", "", 1)
            elif m == 2:
                block = block.replace(f"<arg_value{SUF}>", " <arg_value%s>" % SUF, 1)
            else:
                block += f"<tool_call{SUF}>"
        if r.random() < 0.95:
            block += f"</tool_calls{SUF}>"
        parts.append(block)
        parts.append(r.choice(["", "\n", ctext(r, 2)]))
    return "".join(parts)


def plain_tools(tools):
    return [{"name": t["name"], "parameters": t.get("parameters")} for t in tools] or None


class Reference:
    def __init__(self, tok):
        from vllm.reasoning.hy_v4_reasoning_parser import HYV4ReasoningExtractor
        from vllm.tool_parsers.hy_v4_tool_parser import HYV4ToolExtractor
        self.vocab = tok.get_vocab()
        self.R, self.T = HYV4ReasoningExtractor, HYV4ToolExtractor

    def parse(self, text_, thinking, tools):
        reasoning, content_ = self.R(self.vocab, SUF, thinking).extract_reasoning(text_)
        res = self.T(self.vocab, SUF, True).extract_tool_calls(content_ if content_ is not None else "", tools)
        calls = res["tool_calls"] if res["tools_called"] else []
        if res["tools_called"]:
            content_ = res["content"]
            if content_ and content_.strip() == "":
                content_ = None
        return {"reasoning": reasoning or "", "content": content_ or "",
                "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls]}


def departure(text_, thinking, ref_result):
    """'cut' (no closing </tool_calls>) or 'name' (a call vLLM emits that unholy returns as content), else None."""
    c = text_
    if thinking:
        _, _, c = text_.partition(f"<think{SUF}>") if f"<think{SUF}>" in text_ else ("", "", text_)
        if f"</think{SUF}>" not in c:
            return None
        c = c.partition(f"</think{SUF}>")[2]
    s = c.find(f"<tool_calls{SUF}>")
    if s >= 0 and c.find(f"</tool_calls{SUF}>") == -1:
        return "cut"
    declared = {t["name"] for t in (TOOLS_USED[0] or [])}
    for tc in ref_result["tool_calls"]:
        if tc["name"] not in declared and not re.fullmatch("[A-Za-z0-9_-]{1,64}", tc["name"]):
            return "name"
    return None


TOOLS_USED = [None]


def compare_parse(binary, tok, n, r):
    ref = Reference(tok)
    cases, refs, skipped = [], [], {"cut": 0, "name": 0}
    for _ in range(n):
        tools = [t for t in TOOLS if r.random() < 0.7] if r.random() < 0.8 else []
        TOOLS_USED[0] = tools
        thinking = r.random() < 0.7
        text_ = completion(r, tools, thinking)
        e = ref.parse(text_, thinking, plain_tools(tools))
        d = departure(text_, thinking, e)
        if d:
            skipped[d] += 1
            continue
        wire = [{"type": "function", "function": t} for t in tools]
        cases.append({"text": text_, "thinking": thinking, "tools": wire,
                      "chunk": r.choice([1, 2, 3, 5, 7, 16, 64, 1 << 20])})
        refs.append(e)
    got = run_unholy(binary, "hy4-parse", cases)
    same = 0
    shown = 0
    for c, e, g in zip(cases, refs, got):
        if e == g:
            same += 1
        else:
            shown += 1
            if shown <= 8:
                print(f"MISMATCH parse (thinking={c['thinking']} chunk={c['chunk']}):\n  text     {c['text']!r}\n"
                      f"  tools    {[t['function']['name'] for t in c['tools']]}\n  expected {e}\n  got      {g}")
    calls = sum(1 for e in refs if e["tool_calls"])
    print(f"parse: {same}/{len(cases)} identical ({calls} with calls; {skipped['cut']} cut-off blocks and "
          f"{skipped['name']} invalid names not compared)")
    return same == len(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("ckpt")
    ap.add_argument("--random", type=int, default=10000)
    ap.add_argument("--parse", type=int, default=6000)
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
                    f = c.get("function", c)
                    f["arguments"] = json.dumps(f["arguments"], ensure_ascii=False)
        e = r.choice([None, None, "high", "no_think", "no_think", "low", "auto"])
        if e is not None:
            req["reasoning_effort"] = e
        kw = {}
        if r.random() < 0.15:
            kw["reasoning_effort"] = r.choice(["high", "no_think", "max"])
        if r.random() < 0.15:
            kw["preserved_thinking"] = r.choice([True, False])
        if r.random() < 0.1:
            kw["raw_last_assistant"] = r.choice([True, False])
        if r.random() < 0.1:
            kw["add_vision_id"] = True
        if r.random() < 0.05:
            kw["fallback_strategy"] = "reasoning_toolcall_retry"
        if kw:
            req["chat_template_kwargs"] = kw
        reqs.append(req)
    ok &= compare_render(args.binary, tok, reqs, "random")
    ok &= compare_parse(args.binary, tok, args.parse, r)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

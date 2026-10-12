#!/usr/bin/env python3
"""Compare unholy's tokenizer with HF `tokenizers` on adversarial and real text.

    tok_compare.py <unholy-text-test> <tokenizer.json> [--files F ...] [--random N] [--seed S]

Encodes every string with both (HF: add_special_tokens=False, as vLLM encodes rendered prompts) and
reports mismatches with the first differing position; then decodes random id sequences with both.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer

WS = ["\t", "\n", "\x0b", "\x0c", "\r", " ", "\x1c", "\x1d", "\x1e", "\x1f", "\x85", "\xa0", " ", "᠎",
      " ", " ", " ", " ", " ", " ", " ", "​", "‌", "‍",
      " ", " ", " ", " ", "⁠", "　", "﻿"]
PUNCT = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") + ["。", "，", "、", "「", "」", "…", "—", "–", "·", "¿", "¡",
                                                      "€", "£", "¥", "©", "®", "°", "±", "×", "÷", "→", "∑", "√"]
WORDS = ["hello", "Hello", "HELLO", "don't", "it's", "I'm", "we'll", "x", "a1b2", "CamelCase", "snake_case",
         "über", "naïve", "café", "Ελληνικά", "русский", "עברית", "العربية", "हिन्दी", "ภาษาไทย", "한국어",
         "日本語", "中文字符", "ひらがな", "カタカナ", "ｶﾀｶﾅ", "漢字かな交じり", "é", "ạ̈",
         "👍", "👨‍👩‍👧‍👦", "🇺🇸", "🏳️‍🌈", "1", "12", "123", "1234", "12345", "١٢٣٤", "Ⅻ", "½", "²³", "𝟙𝟚𝟛",
         "0x1F", "3.14159", "1e-5", "-42", "v1.2.3", "http://example.com/a?b=c&d=e", "foo@bar.com",
         "<think>", "</think>", "<｜User｜>", "<｜Assistant｜>", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
         "｜DSML｜", "<｜DSML｜tool_calls>", "<dsml:", "</dsml:", "<｜User", "｜>", "<think", "<<think>>",
         "<｜place▁holder▁no▁7｜>", "<｜deepseek_image｜>", "<|EOT|>", "<|place_holder_mm_span_0437|>",
         "<tool_result>", "\\n", "\\\\", "```python", "def f(x):", "    return x", "\t\tindent", "//", "/*", "*/",
         "#include <stdio.h>", "SELECT * FROM t;", "$$x^2$$", "\\frac{a}{b}", "---", "===", "...", "!!!", "?!",
         "퟿", "", "\U0001f600", "\U00010000", "\U0010fffd", "\x00", "\x01", "\x7f", "\x80", "\x9f"]


def random_string(rng):
    parts = []
    for _ in range(rng.randint(1, 24)):
        r = rng.random()
        if r < 0.30:
            parts.append(rng.choice(WORDS))
        elif r < 0.50:
            parts.append("".join(rng.choice(WS) for _ in range(rng.randint(1, 4))))
        elif r < 0.65:
            parts.append("".join(rng.choice(PUNCT) for _ in range(rng.randint(1, 3))))
        elif r < 0.75:
            parts.append(str(rng.randint(0, 10 ** rng.randint(1, 9))))
        elif r < 0.85:
            # Random code points from the BMP and beyond (skipping surrogates).
            s = []
            for _ in range(rng.randint(1, 6)):
                while True:
                    cp = rng.choice([rng.randint(0x20, 0x2fff), rng.randint(0x3000, 0xd7ff), rng.randint(0xe000, 0xffff),
                                     rng.randint(0x10000, 0x10ffff), rng.randint(0, 0x7f)])
                    if not (0xd800 <= cp < 0xe000):
                        break
                s.append(chr(cp))
            parts.append("".join(s))
        else:
            parts.append(" " * rng.randint(1, 3) + rng.choice(WORDS))
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool")
    ap.add_argument("tokenizer")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--random", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    hf = Tokenizer.from_file(args.tokenizer)

    strings = ["", " ", "  ", "\n", "\n\n", " \n", "a", "Hello world", "Hello  world", "Hello \n world"]
    strings += WORDS + WS + PUNCT
    strings += [w + ws + w2 for w in WORDS[:40] for ws in [" ", "  ", "\n", "　", "\xa0"] for w2 in WORDS[40:60]]
    strings += [ws * k + w for ws in WS for k in (1, 2, 3) for w in ("x", "1", "。", "")]
    strings += ["".join(rng.choice(WS) for _ in range(rng.randint(1, 12))) for _ in range(500)]
    strings += [random_string(rng) for _ in range(args.random)]
    for path in args.files:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        strings.append(text)
        lines = text.splitlines(keepends=True)
        strings += ["".join(lines[i:i + 7]) for i in range(0, len(lines), 7)]

    with tempfile.TemporaryDirectory() as td:
        sp = os.path.join(td, "strings.json")
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(strings, f, ensure_ascii=False)
        ours = json.loads(subprocess.run([args.tool, "tokenize", args.tokenizer, sp], check=True,
                                         capture_output=True).stdout)
    ref = [e.ids for e in hf.encode_batch(strings, add_special_tokens=False)]
    bad = 0
    for s, a, b in zip(strings, ours, ref):
        if a != b:
            bad += 1
            if bad <= 15:
                k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print(f"MISMATCH {s[:120]!r}\n  at {k}: ours {a[max(0, k - 3):k + 5]} "
                      f"{[hf.id_to_token(t) for t in a[k:k + 5]]}\n         hf   {b[max(0, k - 3):k + 5]} "
                      f"{[hf.id_to_token(t) for t in b[k:k + 5]]}")
    ntok = sum(len(b) for b in ref)
    print(f"encode: {len(strings) - bad}/{len(strings)} strings identical ({ntok} tokens)")

    # Decode: random id sequences (including added tokens and partial UTF-8 pieces).
    vocab = hf.get_vocab_size(with_added_tokens=True)
    seqs = [[rng.randrange(vocab) for _ in range(rng.randint(1, 40))] for _ in range(3000)]
    seqs += [b for b in ref[:3000] if b]
    with tempfile.TemporaryDirectory() as td:
        ip = os.path.join(td, "ids.json")
        with open(ip, "w") as f:
            json.dump(seqs, f)
        ours_d = json.loads(subprocess.run([args.tool, "decode", args.tokenizer, ip], check=True,
                                           capture_output=True).stdout)
    bad_d = 0
    for ids, a in zip(seqs, ours_d):
        b = hf.decode(ids, skip_special_tokens=False)
        if a != b:
            bad_d += 1
            if bad_d <= 10:
                print(f"DECODE MISMATCH {ids[:12]}\n  ours {a[:80]!r}\n  hf   {b[:80]!r}")
    print(f"decode: {len(seqs) - bad_d}/{len(seqs)} sequences identical")
    sys.exit(1 if bad or bad_d else 0)


if __name__ == "__main__":
    main()

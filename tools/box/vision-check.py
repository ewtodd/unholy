"""Images through a served model, end to end: a rendered word, colored shapes, an image longer than a step, two images
in one request, and that request repeated (its prompt, past two 256-token pages for both models, should come from the
prefix cache).

    tools/box/model-serve.sh glm vision $GLM_PY tools/box/vision-check.py

URL and MODEL_NAME come from model-serve.sh. Prints each answer, its timing and token counts, and whether it names
what the image shows; exits non-zero when one does not.
"""
import base64
import io
import json
import os
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

URL = os.environ.get("URL", "http://localhost:8300")
MODEL = os.environ.get("MODEL_NAME", "glm53")


def word_image(word, w, h, size):
    """The word in dark letters on a light background, with a blue rectangle and a red disc below it."""
    img = Image.new("RGB", (w, h), (245, 240, 230))
    d = ImageDraw.Draw(img)
    d.rectangle([w * 0.05, h * 0.6, w * 0.3, h * 0.9], fill=(70, 130, 180))
    d.ellipse([w * 0.7, h * 0.6, w * 0.9, h * 0.9], fill=(200, 60, 60))
    d.text((w * 0.1, h * 0.15), word, fill=(20, 20, 20), font=ImageFont.load_default(size=size))
    return img


def shapes_image():
    img = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([80, 180, 320, 420], fill=(220, 30, 30))
    d.rectangle([480, 180, 720, 420], fill=(30, 60, 220))
    return img


def data_url(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def ask(question, images):
    content = [{"type": "image_url", "image_url": {"url": data_url(i)}} for i in images]
    content.append({"type": "text", "text": question})
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 4096,
            "temperature": 0, "reasoning_effort": "low"}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return out["choices"][0]["message"].get("content") or "", out["usage"], time.time() - t


def main():
    pair = [word_image("APPLE", 1280, 720, 192), word_image("BANANA", 1280, 720, 192)]
    two = ("Each image shows one word. Which word is in the first image and which in the second? Answer as "
           "'first: WORD, second: WORD'.")
    cases = [
        ("word", "What word is written in this image? Answer with the word only.",
         [word_image("PELICAN", 640, 360, 96)], [["pelican"]]),
        ("shapes", "Name each shape in this image with its color, left to right, in a few words.", [shapes_image()],
         [["red"], ["circle", "disc", "disk"], ["blue"], ["square", "rectangle"]]),
        # 2560 x 1600: ~5200 tokens, longer than a 3072-token step.
        ("long", "What word is written in this image? Answer with the word only.",
         [word_image("ORCHID", 2560, 1600, 220)], [["orchid"]]),
        ("two", two, pair, [["apple"], ["banana"]]),
        ("repeat", two, pair, [["apple"], ["banana"]]),
    ]
    bad = 0
    first = None
    for name, q, imgs, want in cases:
        answer, usage, dt = ask(q, imgs)
        a = answer.lower()
        ok = all(any(w in a for w in alts) for alts in want)
        if name in ("two", "repeat"):
            ok = ok and a.find("apple") < a.find("banana")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if name == "two":
            first = answer
        if name == "repeat":
            ok = ok and cached > 0 and answer == first
        print(f"{name}: {'ok' if ok else 'WRONG'} in {dt:.1f} s, prompt {usage['prompt_tokens']} tokens "
              f"({cached} cached), completion {usage['completion_tokens']}: {answer.strip()[:200]!r}", flush=True)
        bad += not ok
    print(f"{len(cases)} requests: {'all ok' if bad == 0 else f'{bad} wrong'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

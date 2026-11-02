#!/usr/bin/env python3
"""What a published GGUF build holds, from its shards' headers only (HTTP range requests, no download): bytes and
ggml types per tensor group (routed experts, dense, shared experts, embedding / head, Engram projections, draft
layers past the backbone, vision). A tensor's bytes are the distance to the next tensor's offset.

    python3 tools/ref/gguf_inventory.py REPO PREFIX NSHARDS [--skip-last]

    e.g. unsloth/DeepSeek-V4.1-Flash-1017-GGUF UD-Q2_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL 7 --skip-last

--skip-last leaves out the last shard (the builds that carry their Engram tables as the last shard). Used for
docs/dsv41-flash-quant.md.
"""
import collections
import json
import re
import struct
import sys
import urllib.request

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}

def fetch(url, lo, hi):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (lo, hi)})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

class Buf:
    def __init__(self, url):
        self.url, self.data, self.pos = url, b"", 0
    def need(self, n):
        while self.pos + n > len(self.data):
            more = max(8 << 20, n)
            self.data += fetch(self.url, len(self.data), len(self.data) + more - 1)
    def rd(self, fmt):
        n = struct.calcsize("<" + fmt); self.need(n)
        v = struct.unpack_from("<" + fmt, self.data, self.pos)[0]; self.pos += n; return v
    def rstr(self):
        n = self.rd("Q"); self.need(n); s = self.data[self.pos:self.pos + n]; self.pos += n
        return s.decode("utf-8", "replace")

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
def rval(b, t):
    if t in SC: return b.rd(SC[t])
    if t == 8: return b.rstr()
    if t == 9:
        at = b.rd("I"); n = b.rd("Q")
        return [rval(b, at) for _ in range(n)]
    raise ValueError("kv type %d" % t)

def category(name, n_layers):
    m = re.match(r"blk\.(\d+)\.", name)
    layer = int(m.group(1)) if m else -1
    if "engram" in name: return "engram"
    if "_exps" in name: return "routed experts" if layer < n_layers else "draft/MTP experts"
    if layer >= n_layers: return "draft/MTP dense"
    if "shexp" in name: return "shared experts"
    if name.startswith("token_embd") or name.startswith("output.") or name == "output.weight": return "embed/head"
    if name.startswith("v.") or "vision" in name or name.startswith("mm."): return "vision"
    return "dense (attn, indexer, hc, router, norms)"

repo, prefix, nshards = sys.argv[1], sys.argv[2], int(sys.argv[3])
skip_last = "--skip-last" in sys.argv[4:]
tree = json.load(urllib.request.urlopen("https://huggingface.co/api/models/%s/tree/main?recursive=true" % repo, timeout=60))
sizes = {i["path"]: i.get("lfs", {}).get("size", i.get("size", 0)) for i in tree if i.get("type") == "file"}
by_cat = collections.Counter(); types_by_cat = collections.defaultdict(collections.Counter); meta = {}
n_layers = 1 << 30  # from the first shard's <arch>.block_count
for s in range(1, nshards + 1 - (1 if skip_last else 0)):
    path = "%s-%05d-of-%05d.gguf" % (prefix, s, nshards)
    url = "https://huggingface.co/%s/resolve/main/%s" % (repo, path)
    b = Buf(url)
    assert b.rd("I") == 0x46554747, path
    ver = b.rd("I"); nt = b.rd("Q"); nkv = b.rd("Q")
    align = 32
    for _ in range(nkv):
        k = b.rstr(); t = b.rd("I"); v = rval(b, t)
        if k == "general.alignment": align = v
        if s == 1 and not k.startswith("tokenizer.") and isinstance(v, (int, float, str)) and len(str(v)) < 80:
            meta[k] = v
    if s == 1 and "general.architecture" in meta:
        n_layers = meta.get(meta["general.architecture"] + ".block_count", n_layers)
    infos = []
    for _ in range(nt):
        name = b.rstr(); nd = b.rd("I"); dims = [b.rd("Q") for _ in range(nd)]; typ = b.rd("I"); off = b.rd("Q")
        infos.append((off, name, dims, typ))
    data_start = (b.pos + align - 1) // align * align
    total = sizes[path]
    infos.sort()
    for i, (off, name, dims, typ) in enumerate(infos):
        end = infos[i + 1][0] if i + 1 < len(infos) else total - data_start
        c = category(name, n_layers)
        by_cat[c] += end - off
        types_by_cat[c][TYPES.get(typ, "type%d" % typ)] += end - off
print("==", repo, prefix)
for k in ("general.name", "general.file_type", "general.quantization_version"):
    if k in meta: print("  %s = %s" % (k, meta[k]))
tot = sum(by_cat.values())
for c, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
    mix = ", ".join("%s %.1f" % (t, v / 1e9) for t, v in types_by_cat[c].most_common())
    print("  %-42s %7.2f GB   %s" % (c, n / 1e9, mix))
print("  %-42s %7.2f GB" % ("total read (headers say)", tot / 1e9))
print("  distinct types: %s" % sorted({t for d in types_by_cat.values() for t in d}))

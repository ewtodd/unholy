"""Hy4 flash oracle: transformers' hy_v4_flash (PR huggingface/transformers#50903) on the FP8 release, layer-streamed
on one GPU, teacher-forced.

    /work/env/hy4/bin/python tools/ref/hy4_oracle/run.py --ckpt /scratch/models/hy4-flash-fp8 \
        --ids ids.bin --prompt 1024 --decode 8 --out DIR

The checkpoint (287 GB, 127 GB of it the Engram tables in bf16) does not fit one card, so the text model runs one
decoder layer at a time, as tools/ref/glm_oracle does for GLM: build the layer on the meta device, load its weights
(fp8 block-scaled ones dequantized to bf16(w * s)), push every position through it (the prompt in calls of
--prefill-chunk tokens, then the teacher-forced continuation one token at a time), free it, next layer. An Engram
layer's table stays in host memory (memory-mapped, as transformers' `_no_placement_params` leaves it when a GPU
cannot hold it): only the looked-up rows go to the GPU. The layer code is transformers' own (pinned in
/work/env/hy4); this file does the loading and the model-level glue of HYV4FlashTextModel.forward: the
embedding into hc_mult streams, positions (text: the three M-RoPE planes equal), the causal / sliding-window masks
(built here: HYV4FlashTextModel builds them from the cache of the first layer of each type, which a layer-streamed
run does not have), the indexer's position history, the n-gram tokens and Engram mask
(HYV4FlashModel.build_engram_inputs), then the hyper-head, norm and the fp32 LM head.

Outputs in --out: meta.json, logits.bin (f32 [S, vocab]), h_LLL.bin (bf16 [S, hc_mult, hidden] after layer LLL, for
--dump-layers), routing.json (per MoE layer, how often each expert was picked over all S positions).

--attn-fp32 swaps the eager attention for the same math in fp32 (transformers' eager path runs the softmax in bf16):
the oracle against itself with that change is the noise floor.

--experts-gguf SHARD[,SHARD..] takes the routed experts from a GGUF build of the same weights (blk.L.ffn_{gate,up,down}_exps,
dequantized by gguf-py, e.g. AngelSlim's Q4_K_M or SPARK-4Bit NVFP4): the oracle with a quantization's experts, for KL
against the release.
"""
import argparse
import json
import os
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as hym
from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashConfig

_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8,
}
_NP = {"BF16": np.uint16, "F16": np.uint16, "F32": np.float32, "I32": np.int32, "I64": np.int64, "U8": np.uint8,
       "F8_E4M3": np.uint8}


class Ckpt:
    """Raw sharded-safetensors reader (fp8 included, no safetensors-package dependency)."""

    def __init__(self, d):
        with open(os.path.join(d, "model.safetensors.index.json")) as f:
            wm = json.load(f)["weight_map"]
        self.maps, self.index = {}, {}
        for fn in sorted(set(wm.values())):
            path = os.path.join(d, fn)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            self.maps[fn] = np.memmap(path, dtype=np.uint8, mode="c")
            for k, v in hdr.items():
                if k != "__metadata__":
                    b, e = v["data_offsets"]
                    self.index[k] = (fn, 8 + n + b, 8 + n + e, v["dtype"], v["shape"])

    def has(self, name):
        return name in self.index

    def host(self, name):
        """The tensor in host memory without a copy (memory-mapped, copy-on-write)."""
        fn, b, e, dt, shape = self.index[name]
        a = self.maps[fn][b:e].view(_NP[dt])
        return torch.from_numpy(a).view(_DTYPES[dt]).reshape(shape)

    def raw(self, name, dev):
        fn, b, e, dt, shape = self.index[name]
        t = torch.from_numpy(np.array(self.maps[fn][b:e])).to(dev)
        return t.view(_DTYPES[dt]).reshape(shape)

    def get(self, name, dev, block):
        """A weight as stored, except fp8 + weight_scale_inv (or the stacked experts' `<name>_scale_inv`), which is
        dequantized to bf16(w * s) per 128 x 128 block."""
        t = self.raw(name, dev)
        if t.dtype != torch.float8_e4m3fn:
            return t
        s = self.raw(name + "_scale_inv", dev).float()
        out, inp = t.shape[-2], t.shape[-1]

        def deq(tc, sc):
            sc = sc.repeat_interleave(block[0], -2)[..., :out, :].repeat_interleave(block[1], -1)[..., :inp]
            return (tc.float() * sc).to(torch.bfloat16)

        if t.dim() == 2:
            return deq(t, s)
        res = torch.empty(t.shape, dtype=torch.bfloat16, device=dev)  # stacked experts, in slices
        for e in range(0, t.shape[0], 32):
            res[e:e + 32] = deq(t[e:e + 32], s[e:e + 32])
        return res


# transformers' conversion mapping for hy_v4_flash (checkpoint name -> module name), reversed.
_RENAMES = [
    (".self_attn.sinks", ".self_attn.learnable_sink_param"),
    (".self_attn.gate_proj.weight", ".self_attn.linear_gate.weight"),
    (".post_mlp_layernorm.weight", ".mlp.post_fc2_norm.weight"),
    (".mlp.pre_expert_proj.weight", ".mlp.routed_expert_down_proj.weight"),
    (".mlp.post_expert_proj.weight", ".mlp.routed_expert_up_proj.weight"),
    (".attn_hc.fn", ".hc_attn_layer.hc_pre.hc_fn"),
    (".attn_hc.base", ".hc_attn_layer.hc_pre.hc_base"),
    (".attn_hc.scale", ".hc_attn_layer.hc_pre.hc_scale"),
    (".ffn_hc.fn", ".hc_mlp_layer.hc_pre.hc_fn"),
    (".ffn_hc.base", ".hc_mlp_layer.hc_pre.hc_base"),
    (".ffn_hc.scale", ".hc_mlp_layer.hc_pre.hc_scale"),
]


def checkpoint_name(L, key):
    full = f"model.layers.{L}.{key}"
    for mod, ck in _RENAMES:
        if full.endswith(mod):
            return full[: -len(mod)] + ck
    return full


class GgufExperts:
    """Routed experts of a GGUF build, dequantized to bf16 in the release's layout."""

    def __init__(self, paths):
        import gguf
        from gguf.quants import dequantize
        self.dequantize = dequantize
        self.t = {}
        for p in paths:
            for t in gguf.GGUFReader(p).tensors:
                self.t[t.name] = t

    def get(self, name, dev):
        t = self.t[name]
        a = self.dequantize(t.data, t.tensor_type)  # [experts, out, in] (gguf dims reversed)
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev).to(torch.bfloat16)


def layer_state(layer, L, ck, dev, block, gx=None):
    sd = {}
    for key, p in list(layer.named_parameters()) + list(layer.named_buffers()):
        if key not in layer.state_dict():
            continue  # non-persistent buffer: rebuilt by fix_buffers
        name = checkpoint_name(L, key)
        if key == "engram_layer.multi_head_embedding.embedding.weight":
            sd[key] = ck.host(name)  # stays in host memory
        elif gx is not None and key == "mlp.experts.gate_up_proj":
            sd[key] = torch.cat([gx.get(f"blk.{L}.ffn_gate_exps.weight", dev), gx.get(f"blk.{L}.ffn_up_exps.weight", dev)],
                                dim=1)
        elif gx is not None and key == "mlp.experts.down_proj":
            sd[key] = gx.get(f"blk.{L}.ffn_down_exps.weight", dev)
        else:
            # As stored (assign=True keeps the checkpoint's dtype: bf16, or fp32 where the release keeps fp32,
            # which are the modules transformers keeps in fp32), fp8 dequantized to bf16.
            sd[key] = ck.get(name, dev, block)
    return sd


def fix_buffers(layer, cfg, dev):
    """Non-persistent buffers left on meta by the meta-device construction, rebuilt as _init_weights does."""
    for name, mod in layer.named_modules():
        if isinstance(mod, hym.HYV4FlashRotaryEmbedding):
            for lt, fn in mod.rope_init_fns.items():
                inv, _ = fn(mod.config.per_layer_config[lt], dev, layer_type=lt)
                setattr(mod, f"{lt}_inv_freq", inv.to(dev))
                setattr(mod, f"{lt}_original_inv_freq", inv.clone().to(dev))
        elif isinstance(mod, hym.HYV4FlashEngramEmbedding):
            mod.offsets = mod.head_offsets().to(dev)
        elif isinstance(mod, hym.HYV4FlashEngramLayer):
            mod.head_vocab_sizes = torch.tensor(mod.config_head_vocab_sizes, device=dev)
            compressed = int(mod.lookup_table.max()) + 1
            mod.hash_multipliers = mod.draw_hash_multipliers(compressed).to(dev)
    left = [n for n, b in list(layer.named_parameters()) + list(layer.named_buffers()) if b.is_meta]
    if left:
        raise SystemExit(f"tensors left on meta: {left}")


def eager_attention_fp32(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """hym.eager_attention_forward with the scores, softmax and value sum in fp32."""
    q, k, v = query.float(), key.float(), value.float()
    w = torch.matmul(q, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask.float().clamp(min=-1e30)
    if module.sinks is not None:
        w = torch.cat([w, module.sinks.float().reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)], -1)
    p = torch.softmax(w - w.max(dim=-1, keepdim=True).values, dim=-1)
    if module.sinks is not None:
        p = p[..., :-1]
    o = torch.matmul(p, v).to(value.dtype)
    return o.transpose(1, 2).contiguous(), p


def mask_for(layer_type, s0, s1, window, dev, dtype):
    """The additive mask transformers' create_causal_mask / create_sliding_window_causal_mask gives a call over
    positions [s0, s1) for this layer type's cache: indexed layers hold every position, sliding layers return the
    last window - 1 cached positions plus the call's own."""
    q = torch.arange(s0, s1, device=dev).unsqueeze(1)
    if layer_type == "sliding_attention":
        off = s0 - min(s0, window - 1)
        k = torch.arange(off, s1, device=dev).unsqueeze(0)
        ok = (k <= q) & (k > q - window)
    else:
        k = torch.arange(0, s1, device=dev).unsqueeze(0)
        ok = k <= q
    m = torch.zeros(ok.shape, dtype=dtype, device=dev).masked_fill(~ok, torch.finfo(dtype).min)
    return m[None, None]


def ngram_tokens(ids, n, pad):
    """HYV4FlashTextModel.build_ngram_tokens for one document: [S, n] = (t, t-1, ..., t-n+1), pad before the start."""
    S = ids.shape[0]
    cols = [ids]
    for o in range(1, n):
        prev = torch.full_like(ids, pad)
        prev[o:] = ids[: S - o]
        cols.append(prev)
    return torch.stack(cols, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/models/hy4-flash-fp8")
    ap.add_argument("--ids", required=True, help="uint32 token ids")
    ap.add_argument("--prompt", type=int, required=True, help="prompt length within --ids")
    ap.add_argument("--decode", type=int, default=0, help="teacher-forced decode steps after the prompt")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="push the prompt through in calls of this many "
                    "tokens (0: one call); eager attention takes heads x S x S bf16 scores per call")
    ap.add_argument("--layers", type=int, default=None, help="run only the first N layers (no logits)")
    ap.add_argument("--dump-layers", default="all", help="'all', 'none' or a comma list")
    ap.add_argument("--attn-fp32", action="store_true", help="fp32 scores / softmax (the noise-floor twin)")
    ap.add_argument("--experts-gguf", default=None, help="comma-separated GGUF shards to take the routed experts from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda", 0)
    full = HYV4FlashConfig.from_pretrained(args.ckpt)
    cfg = full.text_config
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"  # transformers' reference per-expert loop
    if args.attn_fp32:
        hym.eager_attention_forward = eager_attention_fp32
    with open(os.path.join(args.ckpt, "config.json")) as f:
        block = json.load(f)["text_config"]["quantization_config"]["weight_block_size"]
    ck = Ckpt(args.ckpt)
    gx = GgufExperts(args.experts_gguf.split(",")) if args.experts_gguf else None
    dt = torch.bfloat16

    ids_all = np.fromfile(args.ids, dtype=np.uint32).astype(np.int64)
    P, S = args.prompt, args.prompt + args.decode
    ids = torch.tensor(ids_all[:S], device=dev)
    n_layers = args.layers if args.layers is not None else cfg.num_hidden_layers
    dump = (set(range(n_layers)) if args.dump_layers == "all" else set()
            if args.dump_layers == "none" else {int(x) for x in args.dump_layers.split(",")})

    with torch.no_grad():
        rope = hym.HYV4FlashRotaryEmbedding(cfg, device=dev)
        embed = ck.get("model.embed_tokens.weight", dev, block)
        h = embed[ids].unsqueeze(0).unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()  # [1, S, hc, hidden]
        del embed
        ngrams = ngram_tokens(ids, cfg.engram_max_ngram_size, cfg.engram_pad_id).unsqueeze(0)  # [1, S, n]
        emask = ((ids != full.image_token_id) & (ids != full.video_token_id) & (ids != cfg.engram_pad_id)).unsqueeze(0)
        cache = DynamicCache(config=cfg)
        pc = args.prefill_chunk or P
        calls = [(s, min(s + pc, P)) for s in range(0, P, pc)] + [(t, t + 1) for t in range(P, S)]
        routing = {}

        for L in range(n_layers):
            t0 = time.time()
            with torch.device("meta"):
                layer = hym.HYV4FlashDecoderLayer(cfg, L)
            layer.load_state_dict(layer_state(layer, L, ck, dev, block, gx), strict=True, assign=True)
            fix_buffers(layer, cfg, dev)
            layer.eval()
            lt = layer.attention_type
            t_load = time.time() - t0

            picks = []
            hook = None
            if isinstance(layer.mlp, hym.HYV4FlashMoE):
                hook = layer.mlp.gate.register_forward_hook(lambda m, i, o: picks.append(o[2].flatten().cpu()))

            outs = []
            for s0, s1 in calls:
                pos = torch.arange(s0, s1, device=dev).view(1, 1, -1).expand(3, 1, -1)
                cos, sin = rope(h, position_ids=pos, layer_type=lt)
                o = layer(
                    h[:, s0:s1],
                    attention_mask=mask_for(lt, s0, s1, cfg.sliding_window, dev, dt),
                    position_ids=pos,
                    cache=cache,
                    use_cache=True,
                    position_embeddings=(cos, sin),
                    indexer_position_ids=torch.arange(0, s1, device=dev).view(1, 1, -1).expand(3, 1, -1),
                    ngram_tokens=ngrams[:, s0:s1],
                    engram_mask=emask[:, s0:s1],
                )
                outs.append(o)
            h = torch.cat(outs, 1)
            if hook is not None:
                hook.remove()
                routing[L] = torch.bincount(torch.cat(picks), minlength=cfg.num_local_experts).tolist()
            del layer, outs
            torch.cuda.empty_cache()
            print(f"layer {L:2d} ({lt:>17s}): load {t_load:5.1f} s  run {time.time() - t0 - t_load:5.1f} s"
                  f"  |h| {h.float().norm():.4e}", flush=True)
            if L in dump:
                h[0].contiguous().view(torch.uint16).cpu().numpy().tofile(os.path.join(args.out, f"h_{L:03d}.bin"))

        meta = {"S": S, "prompt": P, "decode": args.decode, "ids": ids_all[:S].tolist(), "ckpt": args.ckpt,
                "layers": n_layers, "attn_fp32": args.attn_fp32, "experts_gguf": args.experts_gguf,
                "transformers": transformers.__version__,
                "torch": torch.__version__}
        if n_layers == cfg.num_hidden_layers:
            with torch.device("meta"):
                head = hym.HYV4FlashHyperHead(cfg)
                norm = hym.HYV4FlashRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            head.load_state_dict({"hc_fn": ck.get("model.hc_head.hc_head_fn", dev, block).float(),
                                  "hc_base": ck.get("model.hc_head.hc_head_base", dev, block).float(),
                                  "hc_scale": ck.get("model.hc_head.hc_head_scale", dev, block).float()},
                                 strict=True, assign=True)
            norm.load_state_dict({"weight": ck.get("model.norm.weight", dev, block)}, strict=True, assign=True)
            x = norm(head(h))
            w = ck.get("lm_head.weight", dev, block).float()  # the LM head is kept in fp32
            logits = F.linear(x.float(), w)[0]
            logits.cpu().numpy().tofile(os.path.join(args.out, "logits.bin"))
            am = logits.argmax(-1).tolist()
            meta["argmax_after_prompt"] = am[P - 1]
            meta["teacher_forced_argmax"] = am[P - 1:]
            print(f"argmax after prompt: {am[P - 1]}; teacher-forced argmax: {am[P - 1:]}")
        with open(os.path.join(args.out, "routing.json"), "w") as f:
            json.dump(routing, f)
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f)


if __name__ == "__main__":
    main()

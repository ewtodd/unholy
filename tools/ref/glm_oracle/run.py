"""GLM-5.3-Flash oracle: Hugging Face transformers' glm5_next on the FP8 checkpoint, layer-streamed on one GPU,
teacher-forced.

    /work/env/glm/bin/python tools/ref/glm_oracle/run.py --ckpt /scratch/models/glm53-flash-fp8 \
        --ids ids.bin --prompt 1024 --decode 8 --out DIR

The checkpoint (328 GB) does not fit one card, so the text model runs one decoder layer at a time: build the
layer on the meta device, load its weights (fp8 block-scaled ones dequantized to bf16, experts stacked the way
transformers' conversion mapping does), push every position through it (the prompt as one prefill, then the
teacher-forced continuation one token at a time, which exercises the KDA state, the DSA cache and the pooled
indexer's decode path), free it, next layer. The layer code itself is transformers' (pinned commit in
/work/env/glm); this file only does the loading and the model-level glue (embedding into hc_mult streams,
the hyper-head mean, norm, lm_head), mirroring Glm5NextTextModel.forward.

Outputs in --out: meta.json, logits.bin (f32 [S, vocab]), h_LLL.bin (bf16 [S, hc_mult, hidden] after layer
LLL, for --dump-layers), routing.json (per MoE layer, how often each expert was picked over all S positions).
"""
import argparse
import json
import os
import struct
import time

import numpy as np
import torch
import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8,
}
PREFIX = "model.language_model."


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
            self.maps[fn] = np.memmap(path, dtype=np.uint8, mode="r")
            for k, v in hdr.items():
                if k != "__metadata__":
                    b, e = v["data_offsets"]
                    self.index[k] = (fn, 8 + n + b, 8 + n + e, v["dtype"], v["shape"])

    def has(self, name):
        return name in self.index

    def raw(self, name, dev):
        fn, b, e, dt, shape = self.index[name]
        t = torch.from_numpy(np.array(self.maps[fn][b:e])).to(dev)
        return t.view(_DTYPES[dt]).reshape(shape)

    def get(self, name, dev, block):
        """A weight as stored, except fp8 + weight_scale_inv, which is dequantized to bf16."""
        t = self.raw(name, dev)
        if t.dtype != torch.float8_e4m3fn:
            return t
        s = self.raw(name + "_scale_inv", dev).float()
        out, inp = t.shape
        s = s.repeat_interleave(block[0], 0)[:out].repeat_interleave(block[1], 1)[:, :inp]
        return (t.float() * s).to(torch.bfloat16)


def checkpoint_name(key):
    """Layer-module state_dict key -> checkpoint tensor name (transformers' glm5_next renames, reversed)."""
    for gate_part in ("f_a_proj", "f_b_proj", "dt_bias", "A_log"):
        key = key.replace(f"self_attn.forget_gate.{gate_part}", f"self_attn.{gate_part}")
    for side in ("attn", "ffn"):
        for part in ("fn", "base", "scale"):
            if key == f"{side}_hc.{part}":
                return f"hc_{side}_{part}"
    return key


def layer_state(layer, L, ck, dev, block, n_experts):
    pre = f"{PREFIX}layers.{L}."
    sd = {}
    for key in layer.state_dict():
        if key == "mlp.experts.gate_up_proj":
            sd[key] = torch.stack([torch.cat([ck.get(f"{pre}mlp.experts.{e}.gate_proj.weight", dev, block),
                                              ck.get(f"{pre}mlp.experts.{e}.up_proj.weight", dev, block)], 0)
                                   for e in range(n_experts)])
        elif key == "mlp.experts.down_proj":
            sd[key] = torch.stack([ck.get(f"{pre}mlp.experts.{e}.down_proj.weight", dev, block)
                                   for e in range(n_experts)])
        elif key == "self_attn.conv1d.weight":
            sd[key] = torch.cat([ck.get(f"{pre}self_attn.{x}_conv1d.weight", dev, block) for x in "qkv"], 0)
        else:
            sd[key] = ck.get(pre + checkpoint_name(key), dev, block)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/models/glm53-flash-fp8")
    ap.add_argument("--ids", required=True, help="uint32 token ids")
    ap.add_argument("--prompt", type=int, required=True, help="prompt length within --ids")
    ap.add_argument("--decode", type=int, default=0, help="teacher-forced decode steps after the prompt")
    ap.add_argument("--layers", type=int, default=None, help="run only the first N layers (no logits)")
    ap.add_argument("--dump-layers", default="all", help="'all', 'none' or a comma list")
    ap.add_argument("--attn", default="eager", help="transformers attention implementation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda", 0)
    cfg = Glm5NextConfig.from_pretrained(args.ckpt).text_config
    cfg._attn_implementation = args.attn
    cfg._experts_implementation = "eager"  # transformers' reference per-expert loop
    with open(os.path.join(args.ckpt, "config.json")) as f:
        block = json.load(f)["quantization_config"]["weight_block_size"]
    ck = Ckpt(args.ckpt)

    ids_all = np.fromfile(args.ids, dtype=np.uint32).astype(np.int64)
    P, S = args.prompt, args.prompt + args.decode
    ids = torch.tensor(ids_all[:S], device=dev).unsqueeze(0)
    n_layers = args.layers if args.layers is not None else cfg.num_hidden_layers
    dump = (set(range(n_layers)) if args.dump_layers == "all" else set()
            if args.dump_layers == "none" else {int(x) for x in args.dump_layers.split(",")})

    with torch.no_grad():
        embed = ck.get(PREFIX + "embed_tokens.weight", dev, block)
        h = embed[ids].unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()  # [1, S, hc, hidden]
        del embed
        cache = DynamicCache(config=cfg)
        calls = [(0, P)] + [(t, t + 1) for t in range(P, S)]  # (start, end) of each forward call
        topk = [None] * len(calls)
        routing = {}

        for L in range(n_layers):
            t0 = time.time()
            with torch.device("meta"):
                layer = Glm5NextTextDecoderLayer(cfg, L)
            layer.load_state_dict(layer_state(layer, L, ck, dev, block, cfg.num_local_experts), strict=True,
                                  assign=True)
            left = [n for n, b in list(layer.named_parameters()) + list(layer.named_buffers()) if b.is_meta]
            if left:
                raise SystemExit(f"layer {L}: tensors left on meta: {left}")
            layer.eval()
            t_load = time.time() - t0

            picks = []
            hook = None
            if hasattr(layer.mlp, "gate"):
                hook = layer.mlp.gate.register_forward_hook(lambda m, i, o: picks.append(o[2].flatten().cpu()))

            outs = []
            for c, (s0, s1) in enumerate(calls):
                o, tk = layer(
                    h[:, s0:s1],
                    attention_mask=torch.ones(1, s1 - s0, dtype=torch.bool, device=dev),
                    position_ids=torch.arange(s0, s1, device=dev).unsqueeze(0),
                    position_embeddings=None,
                    input_ids=ids[:, s0:s1],
                    past_key_values=cache,
                    prev_topk_indices=topk[c],
                )
                outs.append(o)
                topk[c] = tk  # what Glm5NextTextModel.forward hands the next layer (None after KDA layers)
            h = torch.cat(outs, 1)
            if hook is not None:
                hook.remove()
                routing[L] = torch.bincount(torch.cat(picks), minlength=cfg.num_local_experts).tolist()
            del layer, outs
            torch.cuda.empty_cache()
            print(f"layer {L:2d} ({cfg.layer_types[L]:>25s}): load {t_load:5.1f} s  run {time.time() - t0 - t_load:5.1f} s"
                  f"  |h| {h.float().norm():.4e}", flush=True)
            if L in dump:
                h[0].contiguous().view(torch.uint16).cpu().numpy().tofile(os.path.join(args.out, f"h_{L:03d}.bin"))

        meta = {"S": S, "prompt": P, "decode": args.decode, "ids": ids_all[:S].tolist(), "ckpt": args.ckpt,
                "layers": n_layers, "transformers": transformers.__version__, "torch": torch.__version__}
        if n_layers == cfg.num_hidden_layers:
            x = h.mean(dim=2)  # Glm5NextTextHyperHead: unweighted mean over the hc streams
            w = ck.get(PREFIX + "norm.weight", dev, block)
            xf = x.float()
            x = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)).to(x.dtype) * w
            logits = torch.nn.functional.linear(x, ck.get("lm_head.weight", dev, block)).float()[0]
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

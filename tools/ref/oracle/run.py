"""DSV4 oracle: DeepSeek's own inference/model.py, layer-streamed on one GPU, teacher-forced.

    PYTHONPATH=tools/ref/oracle:/scratch/dsv4-ve/inference $REF_PY tools/ref/oracle/run.py \
        --ckpt /scratch/dsv4-ve --ids ids.bin --prompt 1024 --decode 8 --out DIR

The 168 GB checkpoint does not fit one card, so the model is run one Block at a time: build the
block, load its weights, push every position through it (the prompt as one prefill at start_pos 0,
then the teacher-forced continuation one token at a time, which exercises the decode-path caches),
free it, next layer. The model code itself is unmodified; kernel.py in this directory replaces the
tilelang kernels (see its docstring).

Outputs in --out: meta.json, h_LLL.bin (bf16 [S, 4, 4096] after layer LLL, for --dump-layers),
logits.bin (f32 [S, vocab]).
"""
import argparse
import json
import os
import struct
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "BF16": torch.bfloat16,
    "F16": torch.float16, "F32": torch.float32, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
}


class Ckpt:
    """Raw sharded-safetensors reader (avoids depending on the safetensors package's fp8/e8m0 support)."""

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

    def get(self, name):
        fn, b, e, dt, shape = self.index[name]
        raw = torch.from_numpy(np.array(self.maps[fn][b:e]))
        return raw.view(_DTYPES[dt]).reshape(shape)


def load_params(module, prefix, ck, dev):
    for pname, p in module.named_parameters():
        full = prefix + pname
        if pname.endswith("wo_a.weight"):
            # inference/convert.py: wo_a is dequantized to bf16 (the model runs it as a bf16 einsum).
            w = ck.get(full).to(dev)
            s = ck.get(full[: -len("weight")] + "scale").to(dev)
            wd = (w.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * s[:, None, :, None].float())
            p.data.copy_(wd.flatten(2, 3).flatten(0, 1).to(p.dtype))
            continue
        t = ck.get(full)
        if p.dtype == torch.float4_e2m1fn_x2:
            p.data.view(torch.uint8).copy_(t.view(torch.uint8).to(dev))
        else:
            p.data.copy_(t.to(dev).to(p.dtype))


def install_capture(block, d):
    """Save the prefill-call intermediates of one block (first call only) as raw files: bf16 as
    uint16, fp32/int as is. Names match the engine's UNHOLY_DUMP files."""
    os.makedirs(d, exist_ok=True)
    seen = set()

    def save(name, t):
        if name in seen:
            return
        seen.add(name)
        t = t.detach().contiguous()
        if t.dtype == torch.bfloat16:
            t = t.view(torch.uint16)
        t.cpu().numpy().tofile(os.path.join(d, name + ".bin"))

    orig_pre = block.hc_pre
    calls = {"n": 0}

    def hc_pre(x, fn, scale, base):
        y, post, comb = orig_pre(x, fn, scale, base)
        tag = "attn" if calls["n"] % 2 == 0 else "ffn"
        calls["n"] += 1
        save(f"hcpre_{tag}_y", y[0]); save(f"hcpre_{tag}_post", post[0]); save(f"hcpre_{tag}_comb", comb[0])
        return y, post, comb
    block.hc_pre = hc_pre

    # The attention kernel's exact inputs and output (first call = the prefill).
    import model as M
    orig_attn = M.sparse_attn

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        o = orig_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
        save("sa_q", q[0]); save("sa_kv", kv[0]); save("sa_idx", topk_idxs[0]); save("sa_o", o[0])
        return o
    M.sparse_attn = sparse_attn

    def hook(name, inputs_idx=None):
        def f(mod, inp, out):
            if inputs_idx is not None:
                save(name + "_in", inp[inputs_idx][0])
            if isinstance(out, tuple):
                for i, o in enumerate(out):
                    if torch.is_tensor(o):
                        save(f"{name}_out{i}", o[0] if o.dim() > 2 else o)
            elif torch.is_tensor(out):
                save(name + "_out", out[0] if out.dim() > 2 else out)
        return f
    block.attn.register_forward_hook(hook("attn", 0))
    block.ffn.register_forward_hook(hook("ffn", 0))
    block.ffn.gate.register_forward_hook(hook("gate"))
    block.ffn.shared_experts.register_forward_hook(hook("shared"))
    block.attn.wq_a.register_forward_hook(hook("wq_a"))
    block.attn.wq_b.register_forward_hook(hook("wq_b"))
    block.attn.wkv.register_forward_hook(hook("wkv"))
    block.attn.wo_b.register_forward_hook(hook("wo_b", 0))
    if getattr(block.attn, "compressor", None) is not None:
        block.attn.compressor.register_forward_hook(hook("comp"))
    if getattr(block.attn, "indexer", None) is not None:
        block.attn.indexer.register_forward_hook(hook("indexer"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/dsv4-ve")
    ap.add_argument("--ids", help="uint32 token ids (or --vl-request)")
    ap.add_argument("--vl-request", help="JSON {messages, thinking_mode?}: a prompt with images (data URLs or paths)")
    ap.add_argument("--decode-ids", default="", help="with --vl-request: comma list of teacher-forced continuation ids")
    ap.add_argument("--vision-device", default="", help="run the ViT / Aligner there (e.g. cpu) to see their noise")
    ap.add_argument("--prompt", type=int, default=0, help="prompt length within --ids (--vl-request: computed)")
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--layers", type=int, default=None, help="run only the first N layers (no logits)")
    ap.add_argument("--dump-layers", default="all", help="'all', 'none' or comma list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture-layer", type=int, default=-1, help="save intermediates of this layer's prefill")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    sys.path.insert(1, os.path.join(args.ckpt, "inference"))
    import model as M

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.bfloat16)
    dev = torch.device("cuda", 0)
    torch.set_default_device(dev)

    with open(os.path.join(args.ckpt, "inference", "config.json")) as f:
        cfg = json.load(f)
    images = None
    if args.vl_request:
        # The reference request path: encode_messages -> tokenizer -> prepare_vl_inputs (image blocks).
        sys.path.insert(0, os.path.join(args.ckpt, "encoding"))
        import encoding_dsv4 as E
        import image_processor as ip
        from transformers import AutoTokenizer
        req = json.load(open(args.vl_request))
        prompt, media = E.encode_messages(req["messages"], thinking_mode=req.get("thinking_mode", "chat"),
                                          return_multi_modal_data=True)
        tok = AutoTokenizer.from_pretrained(args.ckpt)
        vargs = types.SimpleNamespace(**cfg)
        with open(os.path.join(args.ckpt, "config.json")) as f:
            for k, v in json.load(f).items():
                if k.startswith("vision_") and not hasattr(vargs, k):
                    setattr(vargs, k, v)
        pids, images = ip.prepare_vl_inputs(prompt, media["images"], tok, vargs)
        dec = [int(x) for x in args.decode_ids.split(",") if x.strip()]
        ids = np.array(list(pids) + dec, dtype=np.int64)
        args.prompt, args.decode = len(pids), len(dec)
    else:
        ids = np.fromfile(args.ids, dtype=np.uint32)[: args.prompt + args.decode].astype(np.int64)
    S, P = len(ids), args.prompt
    margs = M.ModelArgs(**cfg, max_batch_size=1, max_seq_len=((S + 127) // 128) * 128)
    # Normally set by Transformer.__init__.
    M.world_size, M.rank = 1, 0
    M.default_dtype = torch.float8_e4m3fn
    M.scale_fmt = "ue8m0"
    M.scale_dtype = torch.float8_e8m0fnu

    n_layers = args.layers if args.layers is not None else margs.n_layers
    if args.dump_layers == "all":
        dump = set(range(n_layers))
    elif args.dump_layers == "none":
        dump = set()
    else:
        dump = {int(x) for x in args.dump_layers.split(",")}
    os.makedirs(args.out, exist_ok=True)
    ck = Ckpt(args.ckpt)

    input_ids = torch.tensor(ids, dtype=torch.int64, device=dev).unsqueeze(0)
    with torch.inference_mode():
        emb = ck.get("embed.weight").to(dev)
        image_mask = input_ids >= margs.vocab_size
        h = emb[torch.where(image_mask, 0, input_ids)]  # ParallelEmbedding: ids past the vocab embed as 0
        h[image_mask] = 0
        del emb
        visible = None
        if images:
            # merge_image_embeddings with the reference ViT / Aligner, then the prompt's visibility.
            import vision as V
            vdev = torch.device(args.vision_device) if args.vision_device else dev
            ViT, Aligner = V.ViT(margs).to(vdev), V.Aligner(margs).to(vdev)
            for mod, prefix in ((ViT, "vision."), (Aligner, "aligner.")):
                for pname, prm in mod.named_parameters():
                    prm.data.copy_(ck.get(prefix + pname).to(vdev).to(prm.dtype))
            params = torch.stack([ck.get(nm).to(dev) for nm in ("image_start", "image_pad", "image_pad",
                                                                 "image_newline", "image_end")])
            for img in images:
                with torch.device(vdev):
                    emb_i = Aligner(ViT(img.patches.to(vdev), img.n_vit_h, img.n_vit_w), img.n_vit_h, img.n_vit_w)
                emb_i = emb_i.to(dev)[img.perm.to(dev)]
                blk = params[img.types.to(dev)].clone()
                blk[img.types.to(dev) == ip.IMAGE] = emb_i
                h[0, img.start:img.start + blk.size(0)] = blk
            del ViT, Aligner
            visible = M.get_image_visible(input_ids[:, :P], margs.vocab_size, margs.vision_max_n_token)
        h = h.unsqueeze(2).repeat(1, 1, margs.hc_mult, 1)  # [1, S, hc, D] bf16
        for L in range(n_layers):
            t0 = time.time()
            block = M.Block(L, margs)
            load_params(block, f"layers.{L}.", ck, dev)
            if L == args.capture_layer:
                install_capture(block, os.path.join(args.out, f"cap_{L:03d}"))
            t1 = time.time()
            outs = [block(h[:, :P], 0, input_ids[:, :P], visible) if visible is not None else block(h[:, :P], 0, input_ids[:, :P])]
            for t in range(P, S):
                outs.append(block(h[:, t:t + 1], t, input_ids[:, t:t + 1]))
            h = torch.cat(outs, 1)
            torch.cuda.synchronize()
            if L in dump:
                h[0].contiguous().view(torch.uint16).cpu().numpy().tofile(os.path.join(args.out, f"h_{L:03d}.bin"))
            print(f"layer {L:2d}: load {t1 - t0:5.1f} s  run {time.time() - t1:6.1f} s  "
                  f"|h| {h.float().norm().item():.4e}", flush=True)
            del block, outs
            torch.cuda.empty_cache()

        logits_info = None
        if n_layers == margs.n_layers:
            ns = types.SimpleNamespace(norm_eps=margs.norm_eps, hc_eps=margs.hc_eps)
            hc_fn = ck.get("hc_head_fn").to(dev)
            hc_scale = ck.get("hc_head_scale").to(dev)
            hc_base = ck.get("hc_head_base").to(dev)
            x = M.Block.hc_head(ns, h, hc_fn, hc_scale, hc_base)
            norm = M.RMSNorm(margs.dim, margs.norm_eps)
            norm.weight.data.copy_(ck.get("norm.weight").to(dev).float())
            head = M.ParallelHead(margs.vocab_size, margs.dim, margs.norm_eps, margs.hc_eps)
            head.weight.data.copy_(ck.get("head.weight").to(dev).float())
            logits = head(norm(x), full_logits=True)[0]  # [S, V] f32
            logits.cpu().numpy().tofile(os.path.join(args.out, "logits.bin"))
            top = logits.argmax(-1).cpu().tolist()
            logits_info = {"argmax_last_prompt": top[P - 1], "argmax": top,
                           "max_logit_last_prompt": logits[P - 1].max().item()}
            print(f"argmax after prompt: {top[P - 1]} (logit {logits[P - 1].max().item():.3f}); "
                  f"teacher-forced argmax: {top[P - 1:]}", flush=True)

    meta = {"ckpt": args.ckpt, "S": S, "prompt": P, "decode": S - P, "ids": ids.tolist(),
            "layers": n_layers, "dumped": sorted(dump), "logits": logits_info}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()

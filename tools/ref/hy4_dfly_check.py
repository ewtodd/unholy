"""DFly2 (Hy4 flash's drafter) in PyTorch, from vLLM's hy_v4 DFly code as written up in the drafter spec, against
one draft call of unholy's (UNHOLY_HY4_DFLY_DUMP=DIR on any run that drafts, e.g. unholy-hy4-resident-test --spec).

    /work/env/hy4/bin/python tools/ref/hy4_dfly_check.py --ckpt /scratch/models/hy4-flash-fp8 --dump DIR

The dump holds request 0's inputs (anchor, bonus token, the target's final hidden row of the last accepted position,
each draft layer's history: the positions target layer 25 + 4 i recalled there and that layer's cached KV rows) and
unholy's outputs (the final-normed block of 6 rows, the 5 draft tokens and their probabilities). This recomputes the
block and the head chain on the CPU in fp32 with bf16 roundings where the reference rounds, and compares.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_oracle"))
from run import Ckpt  # noqa: E402

BF = torch.bfloat16


def bf(x):
    return x.to(BF).float()


def rms(x, w, eps):
    x = x.float()
    return bf(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float())


def rope(x, pos, theta):
    """Interleaved pairs over the last dim (64): (x[2j], x[2j+1]) rotated by pos * theta^(-2j/64)."""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    ang = pos.double()[:, None] * inv[None, :]
    cos, sin = bf(torch.cos(ang).float()), bf(torch.sin(ang).float())
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [d // 2]
    cos, sin = cos.view(shape), sin.view(shape)
    a, b = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.empty_like(x, dtype=torch.float32)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = a * sin + b * cos
    return bf(out)


def lin(x, w):
    return bf(x.float() @ w.float().t())


def load(path, dtype, shape=None):
    a = np.fromfile(path, dtype=dtype)
    t = torch.from_numpy(a.copy())
    if dtype == np.uint16:
        t = t.view(torch.int16).view(BF).float()
    return t if shape is None else t.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump", required=True)
    a = ap.parse_args()
    with open(os.path.join(a.ckpt, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    eps, D = tc["rms_norm_eps"], tc["hidden_size"]
    with open(os.path.join(a.dump, "meta.json")) as f:
        meta = json.load(f)
    theta = 1e7
    H, dn, dr, dv, R = 32, 192, 64, 256, 512
    W = 6
    anchor, bonus = meta["anchor"], meta["bonus"]
    mask_id = tc.get("dfly_mask_token_id", cfg.get("dfly_mask_token_id"))
    ck = Ckpt(a.ckpt)
    g = lambda n: ck.get(n, "cpu", (128, 128))
    embed = ck.host("model.embed_tokens.weight")
    hidden = load(os.path.join(a.dump, "hidden.bin"), np.uint16, (1, D))
    pos = torch.arange(anchor, anchor + W)

    # slot-0 fusion
    p = "model.dfly_layers."
    x = embed[[bonus] + [mask_id] * (W - 1)].float()
    e = rms(x[:1], g(p + "mtp_fusion.enorm.weight"), eps)
    h = rms(hidden, g(p + "mtp_fusion.hnorm.weight"), eps)
    f = lin(torch.cat([e, h], -1), g(p + "mtp_fusion.eh_proj.weight"))
    x[0] = bf(x[0] + bf(g(p + "mtp_fusion.gate").float() * f[0]))

    resid, mlp = None, None
    for L in range(len(meta["layers"])):
        q_ = f"{p}{L}."
        if L > 0:
            x = bf(mlp + resid)
        resid = x
        hn = rms(x, g(q_ + "input_layernorm.weight"), eps)
        qr = rms(lin(hn, g(q_ + "self_attn.q_a_proj.weight")), g(q_ + "self_attn.q_a_layernorm.weight"), eps)
        q = lin(qr, g(q_ + "self_attn.q_b_proj.weight")).view(W, H, dn + dr)
        q_pe = rope(q[..., dn:], pos, theta)
        kv = lin(hn, g(q_ + "self_attn.kv_a_proj_with_mqa.weight"))
        kv_c = rms(kv[:, :R], g(q_ + "self_attn.kv_a_layernorm.weight"), eps)
        k_pe = rope(kv[:, R:], pos, theta)
        blk = torch.cat([kv_c, k_pe], -1)  # [6, 576]
        gate = lin(hn, g(q_ + "self_attn.linear_gate.weight")).view(W, H, dv)
        kvb = g(q_ + "self_attn.kv_b_proj.weight").float().view(H, dn + dv, R)
        w_uk, w_uv = kvb[:, :dn, :], kvb[:, dn:, :]
        ql = bf(torch.einsum("thp,hpr->thr", q[..., :dn].float(), w_uk))  # [6, H, 512]
        Q = torch.cat([ql, q_pe], -1)  # [6, H, 576]
        hist = load(os.path.join(a.dump, f"hist_{L}.bin"), np.int32)
        hk = load(os.path.join(a.dump, f"kv_{L}.bin"), np.uint16, (hist.numel(), R + dr))
        assert int(hist.max()) < anchor, "history reaches the block"
        sinks = g(q_ + "self_attn.learnable_sink_param").float()
        o = torch.empty(W, H, R)
        for j in range(W):
            K = torch.cat([hk, blk[: j + 1]], 0)  # [n, 576]
            s = (Q[j] @ K.t()) / 16.0  # [H, n]
            s = torch.cat([s, sinks[:, None]], -1)
            pr = torch.softmax(s, -1)[:, :-1]
            o[j] = pr @ K[:, :R]
        o = bf(o)
        ov = bf(torch.einsum("thr,hvr->thv", o, w_uv))  # [6, H, 256]
        ov = bf(ov * bf(torch.sigmoid(gate)))
        att = lin(ov.reshape(W, H * dv), g(q_ + "self_attn.o_proj.weight"))
        att = rms(att, g(q_ + "self_attn.post_proj_norm.weight"), eps)
        x = bf(att + resid)
        resid = x
        hn = rms(x, g(q_ + "post_attention_layernorm.weight"), eps)
        gg = lin(hn, g(q_ + "mlp.gate_proj.weight"))
        uu = lin(hn, g(q_ + "mlp.up_proj.weight"))
        mlp = lin(bf(F.silu(gg) * uu), g(q_ + "mlp.down_proj.weight"))
    out = rms(bf(mlp + resid), g(p + "final_norm.weight"), eps)

    ours = load(os.path.join(a.dump, "x.bin"), np.uint16, (W, D))
    for j in range(W):
        rel = (ours[j] - out[j]).norm() / out[j].norm()
        print(f"block row {j}: rel err {rel:.3e}")

    head = ck.host("lm_head.weight").float()
    hcn, hce = g(p + "hidden_correction.hidden_norm.weight"), g(p + "hidden_correction.embed_norm.weight")
    cg, cu, cd = (g(p + f"hidden_correction.{n}_proj.weight") for n in ("gate", "up", "down"))

    def chain(rows):
        prev, toks, confs = bonus, [], []
        for t in range(W - 1):
            hid = rows[t + 1:t + 2]
            comb = torch.cat([rms(hid, hcn, eps), rms(embed[[prev]].float(), hce, eps)], -1)
            delta = lin(bf(F.silu(lin(comb, cg)) * lin(comb, cu)), cd)
            logits = bf(hid + delta) @ head.t()
            prev = int(logits.argmax())
            toks.append(prev)
            confs.append(round(float(torch.softmax(logits, -1)[0, prev]), 3))
        return toks, confs

    print("reference tokens:", *chain(out))
    print("reference chain on unholy's block:", *chain(ours))
    print("unholy tokens:   ", meta["tokens"], [round(c, 3) for c in meta["conf"]])

if __name__ == "__main__":
    main()

"""DSpark oracle: DeepSeek's own DSparkBlock stages (the checkpoint's mtp.*), teacher-forced, on hidden
states from run.py.

    PYTHONPATH=tools/ref/oracle:/scratch/dsv4-ve/inference $REF_PY tools/ref/oracle/dspark.py \
        --ckpt /scratch/dsv4-ve --ids ids.bin --prompt 1024 --decode 8 --hidden DIR --out DIR

--hidden is a run.py output for the same --ids/--prompt/--decode with layers 40-42 dumped
(--dump-layers 40,41,42): h_LLL.bin, bf16 [S, 4, 4096] after layer LLL. The target's main hidden at a
position is the mean over the 4 HC copies after each target layer, concatenated (model.py
Transformer.forward). This replays the reference speculative loop: forward_spec over the prompt
(start_pos 0: every stage stores the main-KV of the last 128 positions), then for each decode position
t in [P, S - 1): forward_spec(ids[t + 1], main_hidden[t], t) — the block is the token at t + 1 followed
by 4 noise tokens, drafting positions t + 2 .. t + 6 — with temperature 0 (argmax drafts).

Outputs in --out: spec_logits.bin (f32 [D, 5, vocab], markov bias included), spec_ids.bin (int32
[D, 6]: the input token and the 5 drafts), spec_conf.bin (f32 [D, 5]), meta.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import Ckpt, load_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/dsv4-ve")
    ap.add_argument("--ids", required=True, help="uint32 token ids")
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--decode", type=int, required=True)
    ap.add_argument("--hidden", required=True, help="run.py output with h_040..h_042")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(1, os.path.join(args.ckpt, "inference"))
    import model as M

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_default_dtype(torch.bfloat16)
    dev = torch.device("cuda", 0)
    torch.set_default_device(dev)

    with open(os.path.join(args.ckpt, "inference", "config.json")) as f:
        cfg = json.load(f)
    ids = np.fromfile(args.ids, dtype=np.uint32)[: args.prompt + args.decode].astype(np.int64)
    S, P = len(ids), args.prompt
    margs = M.ModelArgs(**cfg, max_batch_size=1, max_seq_len=((S + 127) // 128) * 128)
    margs.temperature = 0.0
    M.world_size, M.rank = 1, 0
    M.default_dtype = torch.float8_e4m3fn
    M.scale_fmt = "ue8m0"
    M.scale_dtype = torch.float8_e8m0fnu
    ck = Ckpt(args.ckpt)

    # main_hidden [1, S, 3 * dim]: mean over the HC copies after each target layer.
    parts = []
    for L in margs.dspark_target_layer_ids:
        h = np.fromfile(os.path.join(args.hidden, f"h_{L:03d}.bin"), dtype=np.uint16)
        h = torch.from_numpy(h.astype(np.int16)).view(torch.bfloat16).reshape(S, margs.hc_mult, margs.dim)
        parts.append(h.to(dev).mean(dim=1))
    main_hidden = torch.cat(parts, dim=-1).unsqueeze(0)

    with torch.inference_mode():
        embed = M.ParallelEmbedding(margs.vocab_size, margs.dim)
        embed.weight.data.copy_(ck.get("embed.weight").to(dev))
        head = M.ParallelHead(margs.vocab_size, margs.dim, margs.norm_eps, margs.hc_eps)
        head.weight.data.copy_(ck.get("head.weight").to(dev).float())
        stages = []
        for i in range(margs.n_mtp_layers):
            blk = M.DSparkBlock(margs.n_layers + i, margs)
            load_params(blk, f"mtp.{i}.", ck, dev)
            blk.embed, blk.head = embed, head
            stages.append(blk)

        def forward_spec(input_ids, mh, start_pos):  # Transformer.forward_spec
            h, main_x = stages[0].forward_embed(mh, input_ids)
            for blk in stages:
                h = blk(h, start_pos, input_ids, main_x)
            if start_pos == 0:
                return None
            return stages[-1].forward_head(h, input_ids)

        ids_t = torch.tensor(ids, dtype=torch.int64, device=dev)
        forward_spec(ids_t[P - 1:P].unsqueeze(0), main_hidden[:, :P], 0)
        logits, out_ids, conf = [], [], []
        for t in range(P, S - 1):
            o, lg, c = forward_spec(ids_t[t + 1:t + 2], main_hidden[:, t:t + 1], t)
            logits.append(lg[0].float().cpu())
            out_ids.append(o[0].int().cpu())
            conf.append(c[0].float().cpu())
            print(f"t={t}: drafts {o[0, 1:].tolist()} conf {[round(x, 3) for x in c[0].float().tolist()]}",
                  flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.stack(logits).numpy().tofile(os.path.join(args.out, "spec_logits.bin"))
    torch.stack(out_ids).numpy().astype(np.int32).tofile(os.path.join(args.out, "spec_ids.bin"))
    torch.stack(conf).numpy().tofile(os.path.join(args.out, "spec_conf.bin"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"prompt": P, "decode": args.decode, "positions": list(range(P, S - 1)),
                   "vocab": margs.vocab_size, "block": margs.dspark_block_size}, f)


if __name__ == "__main__":
    main()

"""Check one Engram layer of unholy against transformers' HYV4FlashEngramLayer on the same input.

    /work/env/hy4/bin/python tools/ref/hy4_oracle/engram_check.py --ckpt /scratch/models/hy4-flash-fp8 \
        --oracle /work/data/hy4/oracle/code1024 --layer 1 --ours DUMP/h_001.bin

Input: the oracle's residual after layer - 1 (h_LLL.bin). transformers' module (CPU, bf16 as the model runs) adds the memory to it; unholy's dump of the same layer run with UNHOLY_HY4_ENGRAM_ONLY=1 and --isolate must
match it at the bf16 level. Also prints the hash ids of a few positions.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as hym
from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashConfig

sys.path.insert(0, os.path.dirname(__file__))
from run import Ckpt, checkpoint_name, fix_buffers, ngram_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/models/hy4-flash-fp8")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--ours", required=True)
    args = ap.parse_args()
    full = HYV4FlashConfig.from_pretrained(args.ckpt)
    cfg = full.text_config
    ck = Ckpt(args.ckpt)
    meta = json.load(open(os.path.join(args.oracle, "meta.json")))
    S, L = meta["S"], args.layer
    hc, D = cfg.hc_mult, cfg.hidden_size
    ids = torch.tensor(meta["ids"], dtype=torch.long)
    h = torch.from_numpy(np.fromfile(os.path.join(args.oracle, f"h_{L - 1:03d}.bin"), dtype=np.uint16).astype(np.int16)
                         ).view(torch.bfloat16).reshape(1, S, hc, D)
    with torch.device("meta"):
        layer = hym.HYV4FlashDecoderLayer(cfg, L)
    eng = layer.engram_layer
    sd = {}
    for key in eng.state_dict():
        name = checkpoint_name(L, "engram_layer." + key)
        sd[key] = ck.host(name) if key == "multi_head_embedding.embedding.weight" else ck.get(name, "cpu", [128, 128])
    eng.load_state_dict(sd, strict=True, assign=True)
    fix_buffers(eng, cfg, "cpu")
    eng.eval()
    ng = ngram_tokens(ids, cfg.engram_max_ngram_size, cfg.engram_pad_id).unsqueeze(0)
    mask = ((ids != full.image_token_id) & (ids != full.video_token_id) & (ids != cfg.engram_pad_id)).unsqueeze(0)
    with torch.no_grad():
        hid = eng.hash_ngrams(ng)
        for t in (0, 1, 5, 593):
            if t < S:
                print(f"pos {t}: tokens {ng[0, t].tolist()} hash ids {hid[0, t].tolist()}")
        ref = h + eng(h, ng, mask)
    ours = torch.from_numpy(np.fromfile(args.ours, dtype=np.uint16).astype(np.int16)).view(torch.bfloat16
                                                                                         ).reshape(1, S, hc, D)
    d = (ours.float() - ref.float())
    rel = d.norm() / ref.float().norm()
    per = (d.pow(2).sum(dim=(2, 3)).sqrt() / ref.float().pow(2).sum(dim=(2, 3)).sqrt())[0]
    print(f"engram layer {L}: rel err {rel:.3e}, worst {per.max():.3e} at pos {int(per.argmax())}; "
          f"engram's own size {(ref.float() - h.float()).norm() / h.float().norm():.3e}")
    added = (ours.float() - h.float())
    want = (ref.float() - h.float())
    print(f"added part: rel err {(added - want).norm() / want.norm():.3e}")





if __name__ == "__main__":
    main()

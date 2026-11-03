"""Plain-PyTorch DFlash2 drafter for GLM-5.3-Flash plus an offline acceptance simulator.

The drafter (`incoai/GLM-5.3-Flash-DFlash2`) is ported from z-lab/dflash's
`Qwen3DFlashAttention`, `Qwen3DFlashDecoderLayer`, `_grouped_dynamic_convolve`,
`GroupedDynamicCausalConv`, `CandidateSelector`, `DFlashDraftModel` /
`DFlash2DraftModel` and `dflash_generate` (with `_attention_mask` semantics),
dropping the transformers/sglang runtime for plain torch.  All five drafter
layers are sliding_attention (window 2048 per query, behind the context too)
and non-causal (`is_causal: false`), so the mask is exactly z-lab's
`_attention_mask` with is_causal=False.

The simulator replays greedy speculative decoding over engine dump directories:
`tokens.i32` (int32 [T]), `features.bf16` (bf16 [T][5][4096], per position the
target hidden state after layers `target_layer_ids` in that order) and
`meta.json` (`prompt_len`, `target_layer_ids`, `label`).

MIT attribution: the ported model code derives from z-lab/dflash
(https://github.com/z-lab/dflash), MIT license.

Usage (env: `nix shell --impure --expr 'with import <nixpkgs> {};
python3.withPackages (p: [p.torch p.safetensors p.numpy])' --command python3 ...`):

    python3 tools/ref/dflash2_ref.py synth OUT_DIR [--T 600] [--prompt-len 300] [--seed 0]
    python3 tools/ref/dflash2_ref.py simulate SEQ_DIR... [--widths 2,3,4,5,6,8]
        [--max-steps N] [--out results.json] [--draft-cost 0.15] [--mtp-accept 0.85]
    python3 tools/ref/dflash2_ref.py selftest [--threads 32]
    python3 tools/ref/dflash2_ref.py bench [--widths 4,8] [--steps 20]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

DEFAULT_DRAFTER_DIR = "/home/e-play/data/unholy/glm53-dflash2"
DEFAULT_GLM_SHARD = "/home/e-play/data/unholy/glm53-fp8-shard1/model-00001-of-00062.safetensors"
EMBED_KEY = "model.language_model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"
HIDDEN = 4096

# Verify cost of a width-B step relative to a one-token step; 5 and 7 interpolate.
VERIFY_COSTS = {1: 1.0, 2: 1.39, 3: 1.74, 4: 2.16, 6: 2.97, 8: 3.59}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def load_config(drafter_dir):
    """Load the drafter's config.json into a dict."""
    with open(os.path.join(drafter_dir, "config.json")) as f:
        return json.load(f)


def _draft_value(config, name, default=None):
    return config.get("dflash_config", {}).get(name, config.get(name, default))


def load_drafter(drafter_dir):
    """Load all drafter tensors from model.safetensors, upcast once to fp32."""
    tensors = {}
    with safe_open(os.path.join(drafter_dir, "model.safetensors"), "pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).float()
    return tensors


def load_glm_embed_head(shard_path):
    """Load GLM's embedding and LM head from one shard, upcast to fp32."""
    out = {}
    with safe_open(shard_path, "pt") as f:
        out["embed"] = f.get_tensor(EMBED_KEY).float()
        out["lm_head"] = f.get_tensor(LM_HEAD_KEY).float()
    return out


def load_sequence(seq_dir):
    """Load a sequence directory: (tokens int64 [T], features fp32 [T, L, H], meta)."""
    tokens = np.fromfile(os.path.join(seq_dir, "tokens.i32"), dtype=np.int32)
    raw = np.fromfile(os.path.join(seq_dir, "features.bf16"), dtype="<u2")
    bits = torch.frombuffer(bytearray(raw.tobytes()), dtype=torch.uint16)
    with open(os.path.join(seq_dir, "meta.json")) as f:
        meta = json.load(f)
    num_layers = len(meta["target_layer_ids"])
    # bf16 bits -> fp32: place each uint16 in the high half of an fp32 word.
    words = bits.to(torch.int32) << 16
    feats = words.view(torch.float32).view(tokens.shape[0], num_layers, HIDDEN)
    assert feats.shape[0] == tokens.shape[0] and feats.shape[1:] == (num_layers, HIDDEN), (feats.shape, tokens.shape)
    return torch.from_numpy(tokens.astype(np.int64)), feats, meta


# ----------------------------------------------------------------------------
# Core math (RMSNorm, RoPE, attention mask)
# ----------------------------------------------------------------------------


def rms_norm(x, weight, eps):
    """RMSNorm (Qwen3 style: x / rms(x) * weight)."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def rotate_half(x):
    """Qwen3-style rotate_half."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions, head_dim, theta):
    """cos/sin tables [len, head_dim] for neox-style RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = positions[:, None].float() * inv_freq[None, :]
    return angles.cos().repeat_interleave(2, dim=-1), angles.sin().repeat_interleave(2, dim=-1)


def apply_rope(x, cos, sin):
    """Apply RoPE to x [heads, len, head_dim] with cos/sin [len, head_dim]."""
    return x * cos[None] + rotate_half(x) * sin[None]


def attention_mask(query_positions, key_positions, sliding_window):
    """Mask for z-lab's `_attention_mask` with is_causal=False: visible iff
    |query_position - key_position| < sliding_window.  [q_len, key_len] bool."""
    qpos = query_positions[:, None]
    kpos = key_positions[None, :]
    return (qpos - kpos < sliding_window) & (kpos - qpos < sliding_window)


def _attention_mask_zlab(key_length, query_length, is_causal, sliding_window):
    """Literal port of z-lab's `_attention_mask` (refs/zlab_dflash_model.py),
    used by selftest to cross-check `attention_mask` above."""
    query_position = key_length - query_length + torch.arange(query_length)[:, None]
    key_position = torch.arange(key_length)[None, :]
    visible = torch.ones((query_length, key_length), dtype=torch.bool)
    if is_causal:
        visible &= key_position <= query_position
    if sliding_window is not None:
        visible &= query_position - key_position < sliding_window
        if not is_causal:
            visible &= key_position - query_position < sliding_window
    return visible


# ----------------------------------------------------------------------------
# Grouped dynamic causal conv (z-lab `_grouped_dynamic_convolve`)
# ----------------------------------------------------------------------------


def grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    """Depthwise grouped conv: tap 0 gets base[0] + dynamic[0], later taps look at
    the `offset`-previous position, zero-padded at the sequence start.  `hidden`
    is [batch, length, hidden] ([length, hidden] gets a batch axis), `dynamic`
    [batch, length, taps, groups], `base` [taps, hidden]."""
    if hidden.dim() == 2:
        hidden = hidden[None]
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)


class GroupedDynamicCausalConv:
    """Port of z-lab's `GroupedDynamicCausalConv.prepare` / `.finish` on plain tensors."""

    def __init__(self, base_kernel, kernel_projection, group_size):
        self.base_kernel = base_kernel  # [2, taps, hidden]
        self.kernel_projection = kernel_projection  # [2*taps*groups, hidden]
        self.group_size = group_size

    def _convolve(self, hidden, dynamic, side):
        return grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[side], self.group_size)

    def prepare(self, hidden):
        """Convolve a sublayer's input; return (convolved, dynamic kernel for finish)."""
        taps = self.base_kernel.shape[1]
        groups = hidden.shape[-1] // self.group_size
        dynamic = F.linear(hidden, self.kernel_projection).view(*hidden.shape[:-1], 2, taps, groups)
        return self._convolve(hidden, dynamic[..., 0, :, :], 0), dynamic[..., 1, :, :]

    def finish(self, hidden, dynamic):
        """Convolve a sublayer's output with the saved dynamic kernel."""
        return self._convolve(hidden, dynamic, 1)


# ----------------------------------------------------------------------------
# Drafter model
# ----------------------------------------------------------------------------


class DFlash2Drafter:
    """DFlash2 draft model over plain tensors: 5 non-causal sliding-attention layers
    with dynamic convs, plus the candidate selector over GLM's LM head."""

    def __init__(self, tensors, config, embed, lm_head):
        self.t = tensors
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_hidden_layers"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.eps = config["rms_norm_eps"]
        self.theta = config["rope_parameters"]["rope_theta"]
        self.window = config["sliding_window"]
        self.block_size = int(_draft_value(config, "block_size", 8))
        self.mask_token_id = _draft_value(config, "mask_token_id")
        self.target_layer_ids = _draft_value(config, "target_layer_ids")
        self.rank = int(_draft_value(config, "selector_rank"))
        self.top_k = int(_draft_value(config, "selector_top_k"))
        self.vocab_size = config["vocab_size"]
        self.embed = embed
        self.lm_head = lm_head
        self.convs = []
        for i in range(self.num_layers):
            self.convs.append((
                GroupedDynamicCausalConv(tensors[f"layers.{i}.attention_conv.base_kernel"],
                                         tensors[f"layers.{i}.attention_conv.kernel_projection.weight"],
                                         _draft_value(config, "conv_group_size", 16)),
                GroupedDynamicCausalConv(tensors[f"layers.{i}.mlp_conv.base_kernel"],
                                         tensors[f"layers.{i}.mlp_conv.kernel_projection.weight"],
                                         _draft_value(config, "conv_group_size", 16)),
            ))
        # Context K/V over the committed prefix, per layer: [layers, len, kv_heads, dim].
        self.context_k = None
        self.context_v = None
        self.context_len = 0

    # -- context keys/values over the committed prefix --------------------------

    def reset_context(self):
        self.context_k = self.context_v = None
        self.context_len = 0

    def _context_kv(self, ctx_hidden, positions):
        """Per-position K (k_proj -> k_norm -> rope) and V (v_proj) from the
        projected context hidden [n, hidden]; returns per-layer lists."""
        k_list, v_list = [], []
        for i in range(self.num_layers):
            k = F.linear(ctx_hidden, self.t[f"layers.{i}.self_attn.k_proj.weight"])
            v = F.linear(ctx_hidden, self.t[f"layers.{i}.self_attn.v_proj.weight"])
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            k = rms_norm(k, self.t[f"layers.{i}.self_attn.k_norm.weight"], self.eps)
            cos, sin = rope_cos_sin(positions, self.head_dim, self.theta)
            k = apply_rope(k.transpose(0, 1), cos, sin).transpose(0, 1)
            k_list.append(k)
            v_list.append(v.view(-1, self.num_kv_heads, self.head_dim))
        return k_list, v_list

    def context_from_features(self, features):
        """Precompute context K/V for all positions at once from [T, L, H] features:
        fc(cat(features)) -> hidden_norm -> per-layer k/v."""
        cat = features.reshape(features.shape[0], -1)
        ctx_hidden = rms_norm(F.linear(cat, self.t["fc.weight"]), self.t["hidden_norm.weight"], self.eps)
        positions = torch.arange(features.shape[0])
        k_list, v_list = self._context_kv(ctx_hidden, positions)
        self.context_k = torch.stack(k_list)
        self.context_v = torch.stack(v_list)
        self.context_len = features.shape[0]

    def extend_context(self, features):
        """Append a [n, L, H] slice of features to the context K/V, position by
        position (equals the all-at-once computation; verified by selftest)."""
        for j in range(features.shape[0]):
            cat = features[j:j + 1].reshape(1, -1)
            ctx_hidden = rms_norm(F.linear(cat, self.t["fc.weight"]), self.t["hidden_norm.weight"], self.eps)
            positions = torch.arange(self.context_len, self.context_len + 1)
            k_list, v_list = self._context_kv(ctx_hidden, positions)
            k_new = torch.stack(k_list).squeeze(1).unsqueeze(1)  # [layers, 1, kv_heads, dim]
            v_new = torch.stack(v_list).squeeze(1).unsqueeze(1)
            if self.context_len == 0:
                self.context_k, self.context_v = k_new, v_new
            else:
                self.context_k = torch.cat([self.context_k, k_new], dim=1)
                self.context_v = torch.cat([self.context_v, v_new], dim=1)
            self.context_len += 1

    # -- one draft block ---------------------------------------------------------

    def block_hidden(self, block_ids, start):
        """Run block ids at positions start..start+B-1 through the drafter; return
        the final normed hidden [B, hidden].  Context K/V must cover positions < start."""
        emb = F.embedding(block_ids, self.embed)
        cos, sin = rope_cos_sin(torch.arange(start, start + block_ids.shape[0]), self.head_dim, self.theta)
        hidden = emb
        for i in range(self.num_layers):
            hidden = self._layer(i, hidden, cos, sin)
        return rms_norm(hidden, self.t["norm.weight"], self.eps)

    def _layer(self, i, hidden, cos, sin):
        """One drafter layer: conv -> attention over context+block keys -> conv -> MLP.
        Accepts [B, hidden] or [1, B, hidden]; returns same rank as input."""
        squeezed = hidden.dim() == 2
        if squeezed:
            hidden = hidden[None]
        t = self.t
        conv_a, conv_m = self.convs[i]
        residual = hidden
        normed = rms_norm(hidden, t[f"layers.{i}.input_layernorm.weight"], self.eps)
        attn_hidden, kernel = conv_a.prepare(normed)

        q = F.linear(attn_hidden, t[f"layers.{i}.self_attn.q_proj.weight"])
        k_blk = F.linear(attn_hidden, t[f"layers.{i}.self_attn.k_proj.weight"])
        v_blk = F.linear(attn_hidden, t[f"layers.{i}.self_attn.v_proj.weight"])
        q = rms_norm(q.view(-1, self.num_heads, self.head_dim),
                     t[f"layers.{i}.self_attn.q_norm.weight"], self.eps)
        k_blk = rms_norm(k_blk.view(-1, self.num_kv_heads, self.head_dim),
                         t[f"layers.{i}.self_attn.k_norm.weight"], self.eps)
        q = apply_rope(q.transpose(0, 1), cos, sin)
        k_blk = apply_rope(k_blk.transpose(0, 1), cos, sin)

        rep = self.num_heads // self.num_kv_heads
        k = torch.cat([self.context_k[i], k_blk.transpose(0, 1)])  # [len+B, kv, dim]
        v = torch.cat([self.context_v[i], v_blk.view(-1, self.num_kv_heads, self.head_dim)])
        k = k.repeat_interleave(rep, dim=1).transpose(0, 1)  # [H, len+B, dim]
        v = v.repeat_interleave(rep, dim=1).transpose(0, 1)

        key_len = k.shape[1]
        q_len = q.shape[1]
        mask = attention_mask(torch.arange(key_len - q_len, key_len),
                              torch.arange(key_len), self.window)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], enable_gqa=True)
        attn_out = attn_out.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        attn_out = F.linear(attn_out, t[f"layers.{i}.self_attn.o_proj.weight"])
        attn_out = conv_a.finish(attn_out, kernel)
        hidden = residual + attn_out

        residual = hidden
        normed = rms_norm(hidden, t[f"layers.{i}.post_attention_layernorm.weight"], self.eps)
        mlp_hidden, kernel = conv_m.prepare(normed)
        gate = F.linear(mlp_hidden, t[f"layers.{i}.mlp.gate_proj.weight"])
        up = F.linear(mlp_hidden, t[f"layers.{i}.mlp.up_proj.weight"])
        mlp_out = F.linear(F.silu(gate) * up, t[f"layers.{i}.mlp.down_proj.weight"])
        mlp_out = conv_m.finish(mlp_out, kernel)
        if squeezed:
            return (residual + mlp_out)[0]
        return residual + mlp_out

    # -- candidate selector ------------------------------------------------------

    def _logits_and_candidates(self, hidden):
        """LM-head logits and top-k candidate ids/values for [n, hidden]."""
        logits = F.linear(hidden, self.lm_head)  # fp32
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        return unary, candidates

    def propose(self, hidden, anchor_id):
        """Greedy draft for [B-1, hidden]: sglang's `build_lattice` — score every
        (slot, top-k predecessor, top-k candidate) edge at once, then walk the
        per-slot argmax map from the anchor; equal to z-lab's sequential greedy
        loop, which `propose_sequential` checks in selftest."""
        unary, candidates = self._logits_and_candidates(hidden)
        hp = F.linear(hidden, self.t["candidate_selector.hidden_projection.weight"])
        cb_p = self.t["candidate_selector.predecessor_codebook"]
        cb_s = self.t["candidate_selector.successor_codebook"]
        n, k = candidates.shape
        anchor_col = torch.full((1, k), int(anchor_id), dtype=torch.long)
        pred_ids = torch.cat([anchor_col, candidates[:-1]], dim=0)  # [n, k]
        pred = cb_p[pred_ids]  # [n, k, rank]
        keys = cb_s[candidates]  # [n, k, rank]
        scores = unary[:, None, :] + torch.einsum(
            "epr,ecr->epc", pred * hp[:, None, :], keys)  # [n, pred, cand]
        maps = scores.argmax(dim=-1)  # [n, k] best candidate per predecessor
        index = int(maps[0, 0].item())  # walk starts from the anchor
        path = []
        for e in range(n):
            path.append(int(candidates[e, index].item()))
            if e + 1 < n:
                index = int(maps[e + 1, index].item())
        return torch.tensor(path, dtype=torch.long)

    def propose_sequential(self, hidden, anchor_id):
        """Reference walk: z-lab's `CandidateSelector.select` greedy loop, scoring
        only the chosen predecessor at each position.  selftest checks this equals
        the batched walk in `propose`."""
        unary, candidates = self._logits_and_candidates(hidden)
        hproj = F.linear(hidden, self.t["candidate_selector.hidden_projection.weight"])
        predecessor = int(anchor_id)
        path = []
        for position in range(hidden.shape[0]):
            scores = unary[position] + self.t["candidate_selector.successor_codebook"][candidates[position]] @ (
                self.t["candidate_selector.predecessor_codebook"][predecessor] * hproj[position])
            index = int(torch.argmax(scores).item())
            predecessor = int(candidates[position, index].item())
            path.append(predecessor)
        return torch.tensor(path, dtype=torch.long)

    # -- one draft step ------------------------------------------------------------

    def draft_step(self, tokens, s, width):
        """Draft `width - 1` tokens from committed position s (context K/V covering
        positions < s): a block of exactly `width` positions, ids [tokens[s],
        mask ...], last width-1 hiddens -> LM head -> greedy selector walk."""
        block_ids = torch.full((width,), self.mask_token_id, dtype=torch.long)
        block_ids[0] = int(tokens[s])
        hidden = self.block_hidden(block_ids, s)[1:]
        return self.propose(hidden, int(tokens[s]))


def timed_draft_step(drafter, tokens, s, width):
    """`draft_step` with wall-clock timing (seconds)."""
    t0 = time.perf_counter()
    drafts = drafter.draft_step(tokens, s, width)
    return drafts, time.perf_counter() - t0


# ----------------------------------------------------------------------------
# Synthetic sequences
# ----------------------------------------------------------------------------


def _fp32_to_bf16_bits(a):
    """Round fp32 numpy to bf16 and return the raw uint16 bit pattern."""
    n = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    n = ((n + np.uint32(0x8000)) >> np.uint32(16)).astype("<u2")
    return torch.from_numpy(n.copy())


def synth_sequence(out_dir, T, prompt_len, seed, target_layer_ids, vocab_size):
    """Write a synthetic sequence: random tokens, Gaussian per-layer features with
    per-layer RMS drawn from U(1, 3), as [T][L][4096] bf16."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab_size, size=T, dtype=np.int32)
    layers = len(target_layer_ids)
    feats = np.empty((T, layers, HIDDEN), dtype=np.float32)
    for li in range(layers):
        feats[:, li] = rng.standard_normal((T, HIDDEN)) * rng.uniform(1.0, 3.0)
    bits = _fp32_to_bf16_bits(feats).numpy().astype("<u2")
    os.makedirs(out_dir, exist_ok=True)
    tokens.tofile(os.path.join(out_dir, "tokens.i32"))
    bits.tofile(os.path.join(out_dir, "features.bf16"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"prompt_len": int(prompt_len),
                   "target_layer_ids": target_layer_ids,
                   "label": f"synthetic seed={seed} T={T} prompt_len={prompt_len}"}, f, indent=2)


# ----------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------


def verify_cost(width):
    """Verify cost for a width-B step (relative to a one-token step), interpolated."""
    if width in VERIFY_COSTS:
        return VERIFY_COSTS[width]
    keys = sorted(VERIFY_COSTS)
    for lo, hi in zip(keys, keys[1:]):
        if lo < width < hi:
            f = (width - lo) / (hi - lo)
            return VERIFY_COSTS[lo] * (1 - f) + VERIFY_COSTS[hi] * f
    raise ValueError(f"no verify cost for width {width}")


def accept_stats(hist, width):
    """From an accepted-length histogram {accepted: count} for width-B drafts:
    (tokens_per_step mean, per-draft-position conditional acceptance list)."""
    total = sum(hist.values())
    mean = (sum(a * n for a, n in hist.items()) + total) / total if total else 0.0
    cond = []
    reached = total  # steps whose draft d-1 chain is intact so far
    for d in range(width - 1):
        accepted = sum(n for a, n in hist.items() if a > d)
        cond.append(accepted / reached if reached else 0.0)
        reached = accepted
    return mean, cond


def simulate_sequence(drafter, tokens, feats, meta, width, max_steps=None):
    """Greedy speculative replay at one width.  Context K/V always covers the
    committed prefix; each step drafts width-1 tokens from a width-sized block and
    accepts the longest prefix matching the target's own greedy continuation."""
    prompt_len = int(meta["prompt_len"])
    drafter.reset_context()
    drafter.context_from_features(feats[:prompt_len])
    s = prompt_len
    last = len(tokens) - 1
    hist = {}
    times = []
    steps = 0
    while s < last and (max_steps is None or steps < max_steps):
        if last - s < width:  # need target tokens s+1..s+width-1 and features s..s+width-1
            break
        drafts, dt = timed_draft_step(drafter, tokens, s, width)
        times.append(dt)
        accepted = 0
        while accepted < width - 1 and int(drafts[accepted]) == int(tokens[s + 1 + accepted]):
            accepted += 1
        hist[accepted] = hist.get(accepted, 0) + 1
        s += accepted + 1
        # commit the bonus token (s_old) and the accepted drafts' features
        drafter.extend_context(feats[s - (accepted + 1):s])
        steps += 1
    mean, cond = accept_stats(hist, width)
    return {"width": width,
            "steps": len(times),
            "tokens_per_step": mean,
            "per_position": cond,
            "histogram": {str(a): n for a, n in sorted(hist.items())},
            "mean_step_time": float(np.mean(times)) if times else 0.0}


def merge_width_stats(rows, width):
    """Merge per-sequence rows of one width into an overall row."""
    hist = {}
    for r in rows:
        for a, n in r["histogram"].items():
            hist[int(a)] = hist.get(int(a), 0) + int(n)
    mean, cond = accept_stats(hist, width)
    steps = sum(r["steps"] for r in rows)
    return {"width": width, "steps": steps, "tokens_per_step": mean,
            "per_position": cond,
            "histogram": {str(a): n for a, n in sorted(hist.items())}}


def speedup_table(overall, draft_cost):
    """(width, tokens/step, verify cost, speedup) rows for the overall results."""
    return [(r["width"], r["tokens_per_step"], verify_cost(r["width"]),
             r["tokens_per_step"] / (verify_cost(r["width"]) + draft_cost))
            for r in sorted(overall, key=lambda r: r["width"])]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_widths(arg):
    return [int(x) for x in arg.split(",") if x.strip()]


def build_drafter():
    """Load config, drafter weights, and GLM's embedding/LM head."""
    config = load_config(DEFAULT_DRAFTER_DIR)
    e = load_glm_embed_head(DEFAULT_GLM_SHARD)
    return DFlash2Drafter(load_drafter(DEFAULT_DRAFTER_DIR), config, e["embed"], e["lm_head"])


def cmd_synth(args):
    config = load_config(DEFAULT_DRAFTER_DIR)
    synth_sequence(args.out_dir, args.T, args.prompt_len, args.seed,
                   _draft_value(config, "target_layer_ids"), config["vocab_size"])
    print(f"synth: wrote {args.out_dir} (T={args.T}, prompt_len={args.prompt_len}, seed={args.seed})")


def cmd_simulate(args):
    torch.set_num_threads(args.threads)
    drafter = build_drafter()
    widths = [w for w in parse_widths(args.widths) if w <= drafter.block_size]
    per_seq, by_width = {}, {}
    for seq_dir in args.seq_dirs:
        tokens, feats, meta = load_sequence(seq_dir)
        name = os.path.basename(os.path.normpath(seq_dir))
        rows = []
        for width in widths:
            row = simulate_sequence(drafter, tokens, feats, meta, width, args.max_steps)
            rows.append(row)
            by_width.setdefault(width, []).append(row)
            print(f"  {name} B={width}: steps={row['steps']} tokens/step={row['tokens_per_step']:.3f} "
                  f"per-position={[f'{p:.2f}' for p in row['per_position']]} "
                  f"hist={row['histogram']} ({row['mean_step_time']*1000:.0f} ms/step)")
        per_seq[name] = rows
    overall = [merge_width_stats(rows, w) for w, rows in sorted(by_width.items())]
    print()
    for row in overall:
        print(f"overall B={row['width']}: steps={row['steps']} tokens/step={row['tokens_per_step']:.3f} "
              f"per-position={[f'{p:.2f}' for p in row['per_position']]}")
    table = speedup_table(overall, args.draft_cost)
    mtp = (1 + args.mtp_accept) / (1.39 + 0.03)
    print(f"\nMTP reference (accept={args.mtp_accept}): (1+{args.mtp_accept})/(1.39+0.03) = {mtp:.3f}")
    print(f"{'B':>3} {'tokens/step':>12} {'verify cost':>12} {'speedup':>9}")
    for w, tps, cost, sp in table:
        print(f"{w:>3} {tps:>12.3f} {cost:>12.2f} {sp:>9.3f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"per_sequence": per_seq, "overall": overall,
                       "speedup": [{"width": w, "tokens_per_step": t, "verify_cost": c,
                                    "speedup": s} for w, t, c, s in table],
                       "mtp_speedup": mtp}, f, indent=2)
        print(f"wrote {args.out}")


def cmd_selftest(args):
    """Check: mask == z-lab `_attention_mask`; per-position context K/V == all-at-once;
    greedy lattice walk == sequential loop over the lattice."""
    torch.set_num_threads(args.threads)
    drafter = build_drafter()

    # 1. attention mask vs z-lab's `_attention_mask`, block of 8 behind 3000 positions.
    q_len, key_len = 8, 3008
    zlab = _attention_mask_zlab(key_len, q_len, False, drafter.window)
    mine = attention_mask(torch.arange(key_len - q_len, key_len),
                          torch.arange(key_len), drafter.window)
    assert torch.equal(zlab, mine), "mask mismatch vs z-lab `_attention_mask`"
    print(f"1. attention_mask == z-lab `_attention_mask` (q_len={q_len}, key_len={key_len}, "
          f"window={drafter.window}, is_causal=False)")

    # 2. per-position context K/V == all-at-once context K/V.
    rng = torch.Generator().manual_seed(0)
    T = 37
    feats = torch.randn(T, len(drafter.target_layer_ids), drafter.hidden_size, generator=rng)
    full = DFlash2Drafter(drafter.t, drafter.config, drafter.embed, drafter.lm_head)
    full.context_from_features(feats)
    inc = DFlash2Drafter(drafter.t, drafter.config, drafter.embed, drafter.lm_head)
    inc.extend_context(feats)
    assert torch.allclose(full.context_k, inc.context_k, atol=1e-4, rtol=1e-4), "context K mismatch"
    assert torch.allclose(full.context_v, inc.context_v, atol=1e-4, rtol=1e-4), "context V mismatch"
    print(f"2. per-position context K/V == all-at-once context K/V over {T} positions, "
          f"{drafter.num_layers} layers (max diff "
          f"{(full.context_k - inc.context_k).abs().max().item():.2e})")

    # 3. greedy batched walk == z-lab's sequential loop over the lattice.
    h = torch.randn(drafter.block_size - 1, drafter.hidden_size, generator=rng)
    fast = drafter.propose(h, drafter.mask_token_id)
    slow = drafter.propose_sequential(h, drafter.mask_token_id)
    assert torch.equal(fast, slow), "selector walk mismatch vs sequential loop"
    print(f"3. greedy lattice walk == sequential loop for {drafter.block_size - 1} draft slots")
    print("selftest passed")


def cmd_bench(args):
    """Time draft steps at the requested widths on the real weights."""
    torch.set_num_threads(args.threads)
    drafter = build_drafter()
    meta = {"prompt_len": 200}
    tokens = torch.arange(154880) % 1000 + 500
    feats = torch.randn(len(tokens), len(drafter.target_layer_ids), drafter.hidden_size)
    for width in parse_widths(args.widths):
        drafter.reset_context()
        drafter.context_from_features(feats[:meta["prompt_len"]])
        timed_draft_step(drafter, tokens, meta["prompt_len"], width)  # warm-up
        times = []
        s = meta["prompt_len"]
        for step in range(args.steps):
            _, dt = timed_draft_step(drafter, tokens, s, width)
            times.append(dt)
            s += 1
            drafter.extend_context(feats[s - 1:s])
            if len(tokens) - s < width:
                drafter.reset_context()
                drafter.context_from_features(feats[:meta["prompt_len"]])
                s = meta["prompt_len"]
        print(f"B={width}: {np.mean(times)*1000:.1f} ms/step over {args.steps} steps "
              f"(min {min(times)*1000:.1f}, max {max(times)*1000:.1f})")


def main():
    parser = argparse.ArgumentParser(description="DFlash2 drafter reference + acceptance simulator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write a synthetic sequence directory")
    p.add_argument("out_dir")
    p.add_argument("--T", type=int, default=600)
    p.add_argument("--prompt-len", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("simulate", help="replay greedy speculative decoding")
    p.add_argument("seq_dirs", nargs="+")
    p.add_argument("--widths", default="2,3,4,5,6,8")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--draft-cost", type=float, default=0.15)
    p.add_argument("--mtp-accept", type=float, default=0.85)
    p.add_argument("--threads", type=int, default=32)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("selftest", help="internal consistency checks")
    p.add_argument("--threads", type=int, default=32)
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("bench", help="time draft steps at given widths")
    p.add_argument("--widths", default="4,8")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--threads", type=int, default=32)
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

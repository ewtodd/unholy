"""Pure-torch stand-ins for DeepSeek-V4's tilelang kernels (inference/kernel.py), for the oracle.

DeepSeek's model.py imports `act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn,
hc_split_sinkhorn` from `kernel`; with this directory first on sys.path it gets these instead. Each
function reproduces the reference kernel's arithmetic, not just its math:

- scales: power-of-two via the same ceil(log2(amax * (1/max))) bit rule (frexp here), amax floors
  1e-4 (fp8) and 6 * 2^-126 (fp4), x / s in fp32, clamp, round-to-nearest-even cast;
- GEMMs: fp32 partial sums per 128-wide (fp8) / 32-wide (fp4) K block, scaled and accumulated in
  fp32, output bf16;
- sparse attention: 64-key blocks with the running-max online softmax, probabilities rounded to bf16
  before the PV product, the attention sink added at the end.

Only the fp32 summation order inside a block differs from the tensor-core kernels.
"""
import os

import torch

# ORACLE_PERTURB=1: sum the GEMM K-blocks in reverse order. Equally valid arithmetic; the KL between a
# perturbed and an unperturbed run is the noise floor any re-implementation is judged against.
_PERTURB = os.environ.get("ORACLE_PERTURB", "0") == "1"

F32 = torch.float32
BF16 = torch.bfloat16

_FP8_MAX_INV = torch.tensor(1.0 / 448.0, dtype=F32).item()
_FP4_MAX_INV = torch.tensor(1.0 / 6.0, dtype=F32).item()
_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(x)) for positive normal fp32 x, exactly (kernel.py fast_round_scale)."""
    m, e = torch.frexp(x)
    e = e - (m == 0.5).to(e.dtype)
    return torch.ldexp(torch.ones_like(x), e)


def _f32_scalar(v: float, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(v, dtype=F32, device=like.device)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt=None, scale_dtype=F32, inplace: bool = False):
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = amax * _f32_scalar(_FP8_MAX_INV, x)
    if scale_fmt is not None:
        s = _pow2_ceil(s)
    q = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2).contiguous(), s.squeeze(-1).to(scale_dtype).contiguous()


def _round_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the e2m1 grid (v already clamped to [-6, 6])."""
    a = v.abs()
    # Ties go to the even code: 0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0, 3.5 -> 4.0, 5 -> 4.0.
    code = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int()
            + (a > 2.5).int() + (a >= 3.5).int() + (a > 5.0).int())
    grid = torch.tensor(_FP4_GRID, dtype=F32, device=v.device)
    return torch.where(v < 0, -grid[code], grid[code])


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=6.0 * 2.0 ** -126)
    s = _pow2_ceil(amax * _f32_scalar(_FP4_MAX_INV, x))
    q = _round_fp4((xf / s).clamp(-6.0, 6.0))
    if inplace:
        x.copy_((q * s).flatten(-2).to(x.dtype))
        return x
    raise NotImplementedError("the oracle only needs the fused quant+dequant form of fp4_act_quant")


def fp8_gemm(a, a_s, b, b_s, scale_dtype=F32):
    """C[M,N] = A[M,K] @ B[N,K]^T, per-128 block scales on both, bf16 out."""
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    A = a.reshape(M, K).float()
    B = b.float()
    sa = a_s.reshape(M, -1).float()
    sb = b_s.float().repeat_interleave(128, dim=0)[:N]  # [N, K/128]
    C = torch.zeros(M, N, dtype=F32, device=a.device)
    for k in (reversed(range(K // 128)) if _PERTURB else range(K // 128)):
        part = A[:, k * 128:(k + 1) * 128] @ B[:, k * 128:(k + 1) * 128].T
        C += part * (sa[:, k:k + 1] * sb[:, k][None, :])
    return C.to(torch.get_default_dtype()).view(*a.shape[:-1], N)


def _unpack_fp4(b: torch.Tensor) -> torch.Tensor:
    """[N, K/2] packed e2m1 (low nibble first) -> [N, K] fp32 grid values."""
    u = b.view(torch.uint8)
    lut = torch.tensor(_FP4_GRID + tuple(-g for g in _FP4_GRID), dtype=F32, device=b.device)
    lo = lut[(u & 0x0F).long()]
    hi = lut[((u >> 4) & 0x0F).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=F32):
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T; A per-128 scales, B per-32 e8m0 scales; bf16 out."""
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    A = a.reshape(M, K).float().view(M, K // 32, 32)
    B = _unpack_fp4(b).view(N, K // 32, 32)
    sa = a_s.reshape(M, -1).float().repeat_interleave(4, dim=1)  # [M, K/32]
    sb = b_s.float()  # [N, K/32]
    part = torch.einsum("mbk,nbk->mnb", A, B)  # fp32 per-32 partial sums
    scaled = part * sa[:, None, :] * sb[None, :, :]
    C = scaled.flip(-1).sum(-1) if _PERTURB else scaled.sum(-1)
    return C.to(torch.get_default_dtype()).view(*a.shape[:-1], N)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk: int = 128):
    """q [b,m,h,d] bf16, kv [b,n,d] bf16, topk_idxs [b,m,topk] int32 (-1 = skip) -> o [b,m,h,d] bf16."""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    block = 64
    o = torch.empty_like(q)
    sink = attn_sink.float()
    for bi in range(b):
        for m0 in range(0, m, chunk):
            m1 = min(m, m0 + chunk)
            qq = q[bi, m0:m1].float()  # [c,h,d]
            idx_all = topk_idxs[bi, m0:m1]
            c = m1 - m0
            acc_o = torch.zeros(c, h, d, dtype=F32, device=q.device)
            sum_exp = torch.zeros(c, h, dtype=F32, device=q.device)
            smax = torch.full((c, h), float("-inf"), dtype=F32, device=q.device)
            for t0 in range(0, topk, block):
                idx = idx_all[:, t0:t0 + block]
                if idx.size(1) < block:
                    pad = torch.full((c, block - idx.size(1)), -1, dtype=idx.dtype, device=idx.device)
                    idx = torch.cat([idx, pad], dim=1)
                valid = idx != -1
                kvb = kv[bi][idx.clamp(min=0).long()]  # [c,block,d] bf16
                kvb = torch.where(valid[..., None], kvb, torch.zeros_like(kvb)).float()
                s = torch.einsum("chd,ckd->chk", qq, kvb)
                s = torch.where(valid[:, None, :], s, torch.full_like(s, float("-inf")))
                s = s * softmax_scale
                prev = smax
                smax = torch.maximum(prev, s.amax(-1))
                scale = torch.exp(prev - smax)
                p = torch.exp(s - smax[..., None])
                sum_exp = sum_exp * scale + p.sum(-1)
                acc_o = acc_o * scale[..., None] + torch.einsum("chk,ckd->chd", p.to(BF16).float(), kvb)
            sum_exp = sum_exp + torch.exp(sink[None, :] - smax)
            o[bi, m0:m1] = (acc_o / sum_exp[..., None]).to(q.dtype)
    return o


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    shape = mixes.shape[:-1]
    hc = hc_mult
    mx = mixes.reshape(-1, (2 + hc) * hc).float()
    pre = torch.sigmoid(mx[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mx[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mx[:, 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).view(-1, hc, hc)
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre.view(*shape, hc), post.view(*shape, hc), comb.view(*shape, hc, hc)

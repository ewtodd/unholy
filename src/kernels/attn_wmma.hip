// Sparse attention with sink on WMMA (gfx12; reference: sparse_attn in kernels/ref_attn.hip).
//
// One 256-thread block per query token computes all 64 heads: DSV4 attention is MQA, so every head
// reads the same KV rows and one block loads each row once. Wave w = (hg, dh) = (w / 2, w % 2) owns
// heads [16 hg, 16 hg + 16) and dims [256 dh, 256 dh + 256): its Q slice lives in registers as WMMA A
// fragments and its output slice in 16 accumulator fragments. Keys go 16 at a time: the rows are staged
// in LDS key-major (the B operand of Q K^T) and dim-major (the B operand of P V); each wave's partial
// scores over its 256 dims are summed with its partner's through LDS; the online softmax runs per head
// with the probabilities rounded to bf16 before P V, as in the reference; the sink joins the
// denominator at the end. The reference blocks the softmax by 64 keys, this by 16: the bf16 rounding
// of P is relative to a different running max, an equally valid order (noise-level difference).
#include "kernels/attn_wmma.hpp"

#include <algorithm>

#include "kernels/dev.hip.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

typedef float float8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ const bf16* paged_row(const PagedRows& p, int r) {
  const int page = r / p.rows_per_page, slot = r % p.rows_per_page;
  return p.pool + (size_t(p.table[page]) * p.rows_per_page + slot) * p.row_stride;
}

constexpr int H = 64, D = 512, KB = 16, NT = 256;

// Query t of one call (AttnKv and the q / pos / o pointers relative to the call).
__device__ __forceinline__ void attn_query(const bf16* __restrict__ q, const int32_t* __restrict__ pos, const AttnKv& kv,
                                           const float* __restrict__ sink, bf16* __restrict__ o, float scale, int t) {
#if defined(UH_GFX12)
  __shared__ alignas(16) uint16_t sK[KB][D + 8];   // [key][dim]
  __shared__ alignas(16) uint16_t sVt[D][KB + 8];  // [dim][key]
  __shared__ float sS[4][2][16][KB + 1];
  __shared__ alignas(16) uint16_t sP[H][KB + 8];
  __shared__ float sF[H], sM[H], sL[H];
  __shared__ const bf16* rowbuf[2][KB];  // the block's rows, and the next block's being looked up
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int hg = wave >> 1, dh = wave & 1;
  const int p_t = pos[t];
  const bool dspark = kv.dspark_slots >= 0;  // DSpark stage: ring slots in order, then the draft block
  const int visible = kv.ratio && !dspark ? (p_t + 1) / kv.ratio : 0;
  const int ncomp = kv.ratio == 0 || dspark ? 0 : (kv.ncomp > 0 ? kv.ncomp : visible);
  const int W = kv.vis ? 512 : 128;  // window entries: positions [lo, hi] from lo up (image spans: 512)
  const int lo = kv.vis ? kv.vis[2 * t] : max(0, p_t - 127), hi = kv.vis ? kv.vis[2 * t + 1] : p_t;
  const int total = dspark ? kv.dspark_slots + kv.dspark_chunk : W + ncomp;

  // Q fragments: head hg*16 + lane%16, dims dh*256 + ks*16 + (lane/16)*8 .. +7.
  bf16x8_t qf[16];
  {
    const bf16* qr = q + (size_t(t) * H + hg * 16 + (lane & 15)) * D + dh * 256 + (lane >> 4) * 8;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) qf[ks] = *reinterpret_cast<const bf16x8_t*>(qr + ks * 16);
  }
  float8_t acc[16];
#pragma unroll
  for (int n = 0; n < 16; ++n)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[n][e] = 0.0f;
  if (tid < H) {
    sM[tid] = -INFINITY;
    sL[tid] = 0.0f;
  }

  // Entry e's row (null: masked). Block b's rows are looked up (top-k index, block table) during block
  // b - 1, by threads 32..47 (wave 1), so the dependent loads overlap the previous block's work.
  const auto row_of = [&](int e) -> const bf16* {
    if (dspark) {
      if (e < kv.dspark_slots) return kv.ring + size_t(e) * D;
      if (e < total) return kv.chunk_kv + size_t(e - kv.dspark_slots) * D;
      return nullptr;
    }
    if (e < W) {
      const int wp = lo + e;
      if (wp <= hi) return wp >= kv.s0 ? kv.chunk_kv + size_t(wp - kv.s0) * D : kv.ring + size_t(wp % 128) * D;
      return nullptr;
    }
    if (e < total) {
      const int ci = e - W;
      const int row = kv.ncomp > 0 ? kv.comp_idx[size_t(t) * kv.ncomp + ci] : ci;
      if (row >= 0 && row < visible) return paged_row(kv.comp, row);
    }
    return nullptr;
  };
  if (tid < KB) rowbuf[0][tid] = row_of(tid);
  __syncthreads();
  for (int e0 = 0, buf = 0; e0 < total; e0 += KB, buf ^= 1) {
    const bf16* const* rows = rowbuf[buf];
    const bool look = tid >= 32 && tid < 32 + KB && e0 + KB < total;  // stored at the end of the block
    const bf16* next = look ? row_of(e0 + KB + tid - 32) : nullptr;
    // Stage 16 rows x 512 dims: unit u -> key u%16, 32 dims at (u/16)*32. Lanes of a half-wave take the 16
    // keys of one dim range, so the transposed 2-byte stores into sVt fall in different banks (with keys
    // per lane and dims 32 apart, all 16 lanes hit one bank: a 16-way conflict on each of the 32 stores).
    {
      const int key = tid & 15, d0 = (tid >> 4) * 32;
      const bf16* r = rows[key];
      uint4 v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] = r ? *reinterpret_cast<const uint4*>(r + d0 + j * 8) : uint4{0, 0, 0, 0};
#pragma unroll
      for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4*>(&sK[key][d0 + j * 8]) = v[j];
      const uint16_t* hv = reinterpret_cast<const uint16_t*>(v);
#pragma unroll
      for (int j = 0; j < 32; ++j) sVt[d0 + j][key] = hv[j];
    }
    __syncthreads();
    // Partial scores over this wave's dims: [16 heads x 16 keys].
    {
      float8_t s;
#pragma unroll
      for (int e = 0; e < 8; ++e) s[e] = 0.0f;
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        const bf16x8_t kf =
            *reinterpret_cast<const bf16x8_t*>(&sK[lane & 15][dh * 256 + ks * 16 + (lane >> 4) * 8]);
        s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(qf[ks], kf, s);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) sS[hg][dh][(lane >> 4) * 8 + e][lane & 15] = s[e];
    }
    __syncthreads();
    // Online softmax: thread -> head tid/4, keys (tid%4)*4 .. +3.
    {
      const int h = tid >> 2, k0 = (tid & 3) * 4, g = h >> 4, hl = h & 15;
      float sv[4], bm = -INFINITY;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int key = k0 + j;
        sv[j] = rows[key] ? (sS[g][0][hl][key] + sS[g][1][hl][key]) * scale : -INFINITY;
        bm = fmaxf(bm, sv[j]);
      }
      bm = fmaxf(bm, __shfl_xor(bm, 1, 32));
      bm = fmaxf(bm, __shfl_xor(bm, 2, 32));
      const float m_old = sM[h];
      const float m_new = fmaxf(m_old, bm);
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float pe = expf(sv[j] - m_new);
        sum += pe;
        sP[h][k0 + j] = f32_to_bf16(pe);
      }
      sum += __shfl_xor(sum, 1, 32);
      sum += __shfl_xor(sum, 2, 32);
      __syncthreads();  // every thread of the head has read sM[h]
      if ((tid & 3) == 0) {
        const float f = expf(m_old - m_new);
        sF[h] = f;
        sL[h] = sL[h] * f + sum;
        sM[h] = m_new;
      }
    }
    __syncthreads();
    // Rescale and P V: A = P [16 heads x 16 keys], B = V^T slice [16 keys x 16 dims] per dim tile.
    {
      const bf16x8_t pf = *reinterpret_cast<const bf16x8_t*>(&sP[hg * 16 + (lane & 15)][(lane >> 4) * 8]);
      float fr[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) fr[e] = sF[hg * 16 + (lane >> 4) * 8 + e];
#pragma unroll
      for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[n][e] *= fr[e];
        const bf16x8_t vf =
            *reinterpret_cast<const bf16x8_t*>(&sVt[dh * 256 + n * 16 + (lane & 15)][(lane >> 4) * 8]);
        acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(pf, vf, acc[n]);
      }
    }
    if (look) rowbuf[buf ^ 1][tid - 32] = next;
    __syncthreads();
  }
  // Normalise with the sink and store: element (head hg*16 + (lane/16)*8 + e, dim dh*256 + n*16 + lane%16).
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int h = hg * 16 + (lane >> 4) * 8 + e;
    const float denom = sL[h] + expf(sink[h] - sM[h]);
    bf16* orow = o + (size_t(t) * H + h) * D + dh * 256 + (lane & 15);
#pragma unroll
    for (int n = 0; n < 16; ++n) orow[n * 16] = f32_to_bf16(acc[n][e] / denom);
  }
#endif
}

__global__ __launch_bounds__(NT) void sparse_attn_wmma_kernel(const bf16* __restrict__ q,
                                                              const int32_t* __restrict__ pos, AttnKv kv,
                                                              const float* __restrict__ sink, bf16* __restrict__ o,
                                                              float scale) {
  attn_query(q, pos, kv, sink, o, scale, blockIdx.x);
}

// Every query of up to AttnBatch::kMax chunks: block -> (chunk, query), then the single-call path.
__global__ __launch_bounds__(NT) void sparse_attn_wmma_batch_kernel(const bf16* __restrict__ q,
                                                                    const int32_t* __restrict__ pos, AttnBatch b,
                                                                    const float* __restrict__ sink,
                                                                    bf16* __restrict__ o, float scale) {
  int c = 0, first = 0;
  while (c + 1 < b.nchunks && int(blockIdx.x) >= first + b.c[c].n) first += b.c[c++].n;
  const AttnChunk ch = b.c[c];
  AttnKv kv;
  kv.chunk_kv = b.kv + size_t(ch.t0) * D;
  kv.ring = b.ring + size_t(ch.slot) * b.ring_stride;
  kv.s0 = ch.s0;
  kv.comp = PagedRows{b.comp_pool, b.tables + size_t(ch.slot) * b.max_pages, b.rows_per_page, b.row_stride};
  kv.comp_idx = b.comp_idx ? b.comp_idx + ch.idx_off : nullptr;
  kv.ncomp = ch.ncomp;
  kv.ratio = b.ratio;
  kv.vis = b.vis ? b.vis + size_t(ch.t0) * 2 : nullptr;
  attn_query(q + size_t(ch.t0) * H * D, pos + ch.t0, kv, sink, o + size_t(ch.t0) * H * D, scale,
             int(blockIdx.x) - first);
}

// ---- Split path: the same per-element arithmetic across more workgroups (decode-sized calls) ----------
// Phase 1: workgroups over (query, group of sb key blocks) compute every head's score of every entry, the
// fused kernel's sv (its two half-dim WMMA chains, their sum, times scale; -inf for a masked entry), into
// S [query][head][ldS], and every entry's row pointer into R [query][ldS] (a masked entry's: a zero row the call's
// first workgroup writes). Scores do not depend on the softmax
// state. Phase 2: sixteen workgroups per query, one per (16-head group, quarter of the dims), each walk the
// entries in the fused kernel's 16-key blocks: the softmax state of their heads (from S, with the fused
// kernel's thread partition, reduction order and roundings) and P V for their 16 heads x 128 dims. Every output
// element sees the same operations in the same order as in the fused kernel.

// A query's entries (the fused kernel's row_of).
struct KeyRange {
  const AttnKv& kv;
  int p_t, visible, W, lo, hi, total;
  bool dspark;
  __device__ KeyRange(const AttnKv& k, const int32_t* pos, int t) : kv(k) {
    p_t = pos[t];
    dspark = kv.dspark_slots >= 0;
    visible = kv.ratio && !dspark ? (p_t + 1) / kv.ratio : 0;
    const int ncomp = kv.ratio == 0 || dspark ? 0 : (kv.ncomp > 0 ? kv.ncomp : visible);
    W = kv.vis ? 512 : 128;
    lo = kv.vis ? kv.vis[2 * t] : max(0, p_t - 127);
    hi = kv.vis ? kv.vis[2 * t + 1] : p_t;
    total = dspark ? kv.dspark_slots + kv.dspark_chunk : W + ncomp;
  }
  __device__ const bf16* row(int e, int t) const {
    if (dspark) {
      if (e < kv.dspark_slots) return kv.ring + size_t(e) * D;
      if (e < total) return kv.chunk_kv + size_t(e - kv.dspark_slots) * D;
      return nullptr;
    }
    if (e < W) {
      const int wp = lo + e;
      if (wp <= hi) return wp >= kv.s0 ? kv.chunk_kv + size_t(wp - kv.s0) * D : kv.ring + size_t(wp % 128) * D;
      return nullptr;
    }
    if (e < total) {
      const int ci = e - W;
      const int row = kv.ncomp > 0 ? kv.comp_idx[size_t(t) * kv.ncomp + ci] : ci;
      if (row >= 0 && row < visible) return paged_row(kv.comp, row);
    }
    return nullptr;
  }
};

__device__ __forceinline__ void attn_scores(const bf16* __restrict__ q, const int32_t* __restrict__ pos,
                                            const AttnKv& kv, float scale, int t, int group, int sb,
                                            float* __restrict__ S, const bf16** __restrict__ R, bf16* __restrict__ zrow,
                                            int ldS) {
#if defined(UH_GFX12)
  __shared__ alignas(16) uint16_t sK[KB][D + 8];
  __shared__ float sS[4][2][16][KB + 1];
  __shared__ const bf16* rows[KB];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int hg = wave >> 1, dh = wave & 1;
  const KeyRange kr(kv, pos, t);
  const int b0 = group * sb, b1 = min(b0 + sb, (kr.total + KB - 1) / KB);
  if (blockIdx.x == 0 && blockIdx.y == 0 && tid < D / 8)
    *reinterpret_cast<uint4*>(zrow + tid * 8) = uint4{0, 0, 0, 0};
  if (b0 >= b1) return;
  bf16x8_t qf[16];
  {
    const bf16* qr = q + (size_t(t) * H + hg * 16 + (lane & 15)) * D + dh * 256 + (lane >> 4) * 8;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) qf[ks] = *reinterpret_cast<const bf16x8_t*>(qr + ks * 16);
  }
  for (int b = b0; b < b1; ++b) {
    const int e0 = b * KB;
    if (tid < KB) {
      rows[tid] = kr.row(e0 + tid, t);
      R[e0 + tid] = rows[tid] ? rows[tid] : zrow;
    }
    __syncthreads();
    {
      const int key = tid & 15, d0 = (tid >> 4) * 32;
      const bf16* r = rows[key];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        *reinterpret_cast<uint4*>(&sK[key][d0 + j * 8]) =
            r ? *reinterpret_cast<const uint4*>(r + d0 + j * 8) : uint4{0, 0, 0, 0};
    }
    __syncthreads();
    {
      float8_t sc;
#pragma unroll
      for (int e = 0; e < 8; ++e) sc[e] = 0.0f;
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        const bf16x8_t kf =
            *reinterpret_cast<const bf16x8_t*>(&sK[lane & 15][dh * 256 + ks * 16 + (lane >> 4) * 8]);
        sc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(qf[ks], kf, sc);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) sS[hg][dh][(lane >> 4) * 8 + e][lane & 15] = sc[e];
    }
    __syncthreads();
    {
      const int h = tid >> 2, k0 = (tid & 3) * 4, g = h >> 4, hl = h & 15;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int key = k0 + j;
        S[size_t(h) * ldS + e0 + key] = rows[key] ? (sS[g][0][hl][key] + sS[g][1][hl][key]) * scale : -INFINITY;
      }
    }
    __syncthreads();
  }
#endif
}

// Phase 2's workgroup: four waves for heads [16 hg, 16 hg + 16) and the quarter's 128 dims, pipelined by key block:
// while waves 0 and 1 run P V of block b (wave w dims [64 w, 64 w + 64)), waves 2 and 3 run block b + 1's softmax
// (the fused kernel's thread partition over their 64 lanes; a head's running max and sum live in registers of its
// four lanes, each computing the fused kernel's values) and all four stage block b + 1's values: double-buffered
// staging, one barrier per block. Block b + 3's values and scores load during step b into the register set block b
// left (the step is unrolled by three, so no register waits at its end), and the barriers do not wait for them.
constexpr int NT2 = 128;
__device__ __forceinline__ void attn_pv_tile(const int32_t* __restrict__ pos, const AttnKv& kv,
                                             const float* __restrict__ sink, bf16* __restrict__ o, int t, int hg,
                                             int dq, const float* __restrict__ S, const bf16* const* __restrict__ R,
                                             int ldS, const float2* __restrict__ rope) {
#if defined(UH_GFX12)
  __shared__ alignas(16) uint16_t sVt[2][128][KB + 8];  // [dim of the quarter][key]
  __shared__ alignas(16) uint16_t sP[2][16][KB + 8];
  __shared__ float sF[2][16], sM[16], sL[16];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const bool pv = wave < 2;
  // The output rope (rope_kernel, inverse) on dims [448, 512): wave 1 of the last quarter. Its lanes' frequencies.
  const bool roped = rope && dq == 3 && wave == 1;
  float2 rf[4] = {};
  const KeyRange kr(kv, pos, t);
  const int nb = (kr.total + KB - 1) / KB;
  const int key = tid & 15, d0 = (tid >> 4) * 16;  // V^T staging: key, 16 dims of the quarter
  const int hl = (tid >> 2) & 15, k0 = (tid & 3) * 4;  // softmax (waves 2, 3): head 16 hg + hl, keys k0 .. k0 + 3
  const float* Sh = S + size_t(hg * 16 + hl) * ldS + k0;
  float8_t acc[4];
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[n][e] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  float sk[8];  // the output lanes' heads' sinks, loaded ahead
#pragma unroll
  for (int e = 0; e < 8; ++e) sk[e] = sink[hg * 16 + (lane >> 4) * 8 + e];
  if (roped)
#pragma unroll
    for (int n = 0; n < 4; ++n) rf[n] = rope[size_t(kr.p_t) * 32 + ((n * 16 + (lane & 15)) >> 1)];
  // Every step issues the same loads without branches or uses right after them (the wait counts stay exact):
  // past the last block they repeat it.
  const auto blk = [&](int b) { return min(b, nb - 1) * KB; };
  const auto load = [&](const bf16* r, int b, u32x4_t (&v)[2], float4& s) {
#pragma unroll
    for (int j = 0; j < 2; ++j) v[j] = *as_global(reinterpret_cast<const u32x4_t*>(r + dq * 128 + d0 + j * 8));
    s = *reinterpret_cast<const float4*>(Sh + blk(b));
  };
  // Block b's values into staging buffer b & 1; waves 2, 3: its softmax.
  const auto stage = [&](int b, const u32x4_t (&v)[2], const float4& sc) {
    const int buf = b & 1;
    const uint16_t* hv = reinterpret_cast<const uint16_t*>(v);
#pragma unroll
    for (int j = 0; j < 16; ++j) sVt[buf][d0 + j][key] = hv[j];
    if (!pv) {  // online softmax: the fused kernel's arithmetic, scores from S
      const float sv[4] = {sc.x, sc.y, sc.z, sc.w};
      float bm = -INFINITY;
#pragma unroll
      for (int j = 0; j < 4; ++j) bm = fmaxf(bm, sv[j]);
      bm = fmaxf(bm, xor_lane<1>(bm));
      bm = fmaxf(bm, xor_lane<2>(bm));
      const float m_new = fmaxf(m, bm);
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float pe = expf(sv[j] - m_new);
        sum += pe;
        sP[buf][hl][k0 + j] = f32_to_bf16(pe);
      }
      sum += xor_lane<1>(sum);
      sum += xor_lane<2>(sum);
      const float f = expf(m - m_new);
      if ((tid & 3) == 0) sF[buf][hl] = f;
      l = l * f + sum;
      m = m_new;
    }
  };
  // Waves 0, 1: P V of block b.
  const auto pv_block = [&](int b) {
    if (!pv) return;
    const int buf = b & 1;
    const bf16x8_t pf = *reinterpret_cast<const bf16x8_t*>(&sP[buf][lane & 15][(lane >> 4) * 8]);
    float fr[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) fr[e] = sF[buf][(lane >> 4) * 8 + e];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[n][e] *= fr[e];
      const bf16x8_t vf =
          *reinterpret_cast<const bf16x8_t*>(&sVt[buf][wave * 64 + n * 16 + (lane & 15)][(lane >> 4) * 8]);
      acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(pf, vf, acc[n]);
    }
  };
  u32x4_t v0[2], v1[2], v2[2];
  float4 s0, s1, s2;
  load(R[key], 0, v0, s0);
  load(R[blk(1) + key], 1, v1, s1);
  load(R[blk(2) + key], 2, v2, s2);
  const bf16* rp = R[blk(3) + key];  // block b + 3's row at step b
  stage(0, v0, s0);
  // Step b (b + 1 < nb): block b + 3's loads into block b's set, barrier, P V of b, block b + 1's staging.
  const auto step = [&](int b, u32x4_t (&vl)[2], float4& sl, const u32x4_t (&vs)[2], const float4& ss) {
    const bf16* rq = R[blk(b + 4) + key];  // first: waiting for it waits for nothing issued later
    load(rp, b + 3, vl, sl);
    rp = rq;
    lds_barrier();
    pv_block(b);
    stage(b + 1, vs, ss);
  };
  int b = 0;
  for (; b + 4 <= nb; b += 3) {
    step(b, v0, s0, v1, s1);
    step(b + 1, v1, s1, v2, s2);
    step(b + 2, v2, s2, v0, s0);
  }
  if (b + 1 < nb) step(b, v0, s0, v1, s1);
  if (b + 2 < nb) step(b + 1, v1, s1, v2, s2);
  lds_barrier();
  pv_block(nb - 1);
  if (!pv && (tid & 3) == 0) {
    sM[hl] = m;
    sL[hl] = l;
  }
  lds_barrier();
  if (!pv) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int h = (lane >> 4) * 8 + e;
    const float denom = sL[h] + expf(sk[e] - sM[h]);
    bf16* orow = o + (size_t(t) * H + hg * 16 + h) * D + dq * 128 + wave * 64 + (lane & 15);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const float v = acc[n][e] / denom;
      if (roped) {  // the pair (even dim, odd dim) is (lane, lane ^ 1), as rope_kernel reads it back from bf16
        const float x = round_bf16(v), y = xor_lane<1>(x);
        const bool odd = lane & 1;
        const float a = odd ? y : x, b2 = odd ? x : y, c = rf[n].x, sn = -rf[n].y;
        // rope_kernel's a * c - b * s and a * s + b * c as it compiles: fma(a, c, -(b * s)), fma(a, s, b * c).
        const float r0 = __builtin_fmaf(a, c, -__fmul_rn(b2, sn)), r1 = __builtin_fmaf(a, sn, __fmul_rn(b2, c));
        orow[n * 16] = f32_to_bf16(odd ? r1 : r0);
      } else {
        orow[n * 16] = f32_to_bf16(v);
      }
    }
  }
#endif
}

__global__ __launch_bounds__(NT) void attn_scores_kernel(const bf16* __restrict__ q, const int32_t* __restrict__ pos,
                                                         AttnKv kv, float scale, int sb, float* __restrict__ S,
                                                         const bf16** __restrict__ R, bf16* __restrict__ zrow,
                                                         int ldS) {
  attn_scores(q, pos, kv, scale, blockIdx.x, blockIdx.y, sb, S + size_t(blockIdx.x) * H * ldS,
              R + size_t(blockIdx.x) * ldS, zrow, ldS);
}
__global__ __launch_bounds__(NT2) void attn_pv_kernel(const int32_t* __restrict__ pos, AttnKv kv,
                                                      const float* __restrict__ sink, bf16* __restrict__ o,
                                                      const float* __restrict__ S, const bf16* const* __restrict__ R,
                                                      int ldS) {
  attn_pv_tile(pos, kv, sink, o, blockIdx.x, blockIdx.y >> 2, blockIdx.y & 3, S + size_t(blockIdx.x) * H * ldS,
               R + size_t(blockIdx.x) * ldS, ldS, nullptr);
}

// The batch's chunk of block x (as sparse_attn_wmma_batch_kernel): its AttnKv, first row and the local query.
__device__ __forceinline__ AttnKv chunk_kv(const AttnBatch& b, int& t0, int& tl, int* n = nullptr) {
  int c = 0, first = 0;
  while (c + 1 < b.nchunks && int(blockIdx.x) >= first + b.c[c].n) first += b.c[c++].n;
  const AttnChunk ch = b.c[c];
  AttnKv kv;
  kv.chunk_kv = b.kv + size_t(ch.t0) * D;
  kv.ring = b.ring + size_t(ch.slot) * b.ring_stride;
  kv.s0 = ch.s0;
  kv.comp = PagedRows{b.comp_pool, b.tables + size_t(ch.slot) * b.max_pages, b.rows_per_page, b.row_stride};
  kv.comp_idx = b.comp_idx ? b.comp_idx + ch.idx_off : nullptr;
  kv.ncomp = ch.ncomp;
  kv.ratio = b.ratio;
  kv.vis = b.vis ? b.vis + size_t(ch.t0) * 2 : nullptr;
  t0 = ch.t0;
  tl = int(blockIdx.x) - first;
  if (n) *n = ch.n;
  return kv;
}
__global__ __launch_bounds__(NT) void attn_scores_batch_kernel(const bf16* __restrict__ q,
                                                               const int32_t* __restrict__ pos, AttnBatch b,
                                                               float scale, int sb, float* __restrict__ S,
                                                               const bf16** __restrict__ R, bf16* __restrict__ zrow,
                                                               int ldS) {
  int t0 = 0, tl = 0, n = 0;
  const AttnKv kv = chunk_kv(b, t0, tl, &n);
  // A one-row chunk's window update (window_update_kernel): its slot (s0 % 128) is no entry of its query.
  if (b.ring_out && n == 1 && blockIdx.y == 0 && threadIdx.x < D / 8)
    *reinterpret_cast<uint4*>(b.ring_out + (kv.ring - b.ring) + size_t(kv.s0 % 128) * D + threadIdx.x * 8) =
        *reinterpret_cast<const uint4*>(kv.chunk_kv + threadIdx.x * 8);
  attn_scores(q + size_t(t0) * H * D, pos + t0, kv, scale, tl, blockIdx.y, sb, S + size_t(blockIdx.x) * H * ldS,
              R + size_t(blockIdx.x) * ldS, zrow, ldS);
}
__global__ __launch_bounds__(NT2) void attn_pv_batch_kernel(const int32_t* __restrict__ pos, AttnBatch b,
                                                            const float* __restrict__ sink, bf16* __restrict__ o,
                                                            const float* __restrict__ S,
                                                            const bf16* const* __restrict__ R, int ldS) {
  int t0 = 0, tl = 0;
  const AttnKv kv = chunk_kv(b, t0, tl);
  attn_pv_tile(pos + t0, kv, sink, o + size_t(t0) * H * D, tl, blockIdx.y >> 2, blockIdx.y & 3,
               S + size_t(blockIdx.x) * H * ldS, R + size_t(blockIdx.x) * ldS, ldS, b.rope);
}

}  // namespace

void sparse_attn_wmma_batch(const bf16* q, const int32_t* pos, const AttnBatch& b, const float* sink, bf16* o,
                            float scale, hipStream_t st) {
  if (b.nchunks <= 0 || b.nchunks > AttnBatch::kMax) throw std::runtime_error("sparse_attn_wmma_batch: chunk count");
  int n = 0;
  for (int i = 0; i < b.nchunks; ++i) n += b.c[i].n;
  if (n == 0) return;
  sparse_attn_wmma_batch_kernel<<<n, NT, 0, st>>>(q, pos, b, sink, o, scale);
  UH_CHECK_LAUNCH();
}

void sparse_attn_wmma(const bf16* q, const int32_t* pos, AttnKv kv, const float* sink, bf16* o, int T, int heads,
                      int dim, float scale, hipStream_t st) {
  if (heads != H || dim != D) throw std::runtime_error("sparse_attn_wmma: 64 heads x 512 only");
  if (T <= 0) return;
  sparse_attn_wmma_kernel<<<T, NT, 0, st>>>(q, pos, kv, sink, o, scale);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

namespace unholy::kernels {

namespace {
// Phase 1's key blocks per workgroup: about 128 workgroups for the call, at most 8 blocks each.
int scores_sb(int n, int ldS) { return std::clamp((n * (ldS / KB) + 127) / 128, 1, 8); }
// S, then R ([n][ldS] row pointers) and the zero row in the caller's buffer: false when it is too small.
bool split_bufs(float* S, size_t s_floats, int n, int ldS, const bf16**& R, bf16*& zrow) {
  const size_t sf = size_t(n) * H * ldS, rf = size_t(n) * ldS * (sizeof(void*) / 4);
  if (sf + rf + D / 2 > s_floats) return false;
  R = reinterpret_cast<const bf16**>(S + sf);
  zrow = reinterpret_cast<bf16*>(S + sf + rf);
  return true;
}
}  // namespace

bool sparse_attn_wmma_split(const bf16* q, const int32_t* pos, AttnKv kv, const float* sink, bf16* o, int T, float scale,
                            int max_total, float* S, size_t s_floats, hipStream_t st) {
  if (T <= 0) return true;
  const int ldS = (max_total + KB - 1) / KB * KB;
  const bf16** R = nullptr;
  bf16* zrow = nullptr;
  if (kv.dspark_slots >= 0 || !split_bufs(S, s_floats, T, ldS, R, zrow)) return false;
  const int sb = scores_sb(T, ldS), groups = (ldS / KB + sb - 1) / sb;
  attn_scores_kernel<<<dim3(T, groups), NT, 0, st>>>(q, pos, kv, scale, sb, S, R, zrow, ldS);
  attn_pv_kernel<<<dim3(T, 16), NT2, 0, st>>>(pos, kv, sink, o, S, R, ldS);
  UH_CHECK_LAUNCH();
  return true;
}

bool sparse_attn_wmma_batch_split(const bf16* q, const int32_t* pos, const AttnBatch& b, const float* sink, bf16* o,
                                  float scale, int max_total, float* S, size_t s_floats, hipStream_t st) {
  int n = 0;
  for (int i = 0; i < b.nchunks; ++i) n += b.c[i].n;
  if (n == 0) return true;
  const int ldS = (max_total + KB - 1) / KB * KB;
  const bf16** R = nullptr;
  bf16* zrow = nullptr;
  if (!split_bufs(S, s_floats, n, ldS, R, zrow)) return false;
  const int sb = scores_sb(n, ldS), groups = (ldS / KB + sb - 1) / sb;
  attn_scores_batch_kernel<<<dim3(n, groups), NT, 0, st>>>(q, pos, b, scale, sb, S, R, zrow, ldS);
  attn_pv_batch_kernel<<<dim3(n, 16), NT2, 0, st>>>(pos, b, sink, o, S, R, ldS);
  UH_CHECK_LAUNCH();
  return true;
}

}  // namespace unholy::kernels

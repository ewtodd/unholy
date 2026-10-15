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
#include "kernels/dev.hip.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

typedef float float8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));

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
// Phase 1: workgroups over (query, group of SB key blocks) compute every head's score of every entry, the
// fused kernel's sv (its two half-dim WMMA chains, their sum, times scale; -inf for a masked entry), into
// S [query][head][ldS]. Scores do not depend on the softmax state. Phase 2: four workgroups per query, one
// per quarter of the dims, each walk the entries in the fused kernel's 16-key blocks: the softmax state of
// all 64 heads (from S, with the fused kernel's thread partition, reduction order and roundings: identical
// in all four) and P V for their quarter of the output. Every output element sees the same operations in
// the same order as in the fused kernel.
constexpr int SB = 8;  // key blocks per phase-1 workgroup

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
                                            const AttnKv& kv, float scale, int t, int group, float* __restrict__ S,
                                            int ldS) {
#if defined(UH_GFX12)
  __shared__ alignas(16) uint16_t sK[KB][D + 8];
  __shared__ float sS[4][2][16][KB + 1];
  __shared__ const bf16* rows[KB];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int hg = wave >> 1, dh = wave & 1;
  const KeyRange kr(kv, pos, t);
  const int b0 = group * SB, b1 = min(b0 + SB, (kr.total + KB - 1) / KB);
  if (b0 >= b1) return;
  bf16x8_t qf[16];
  {
    const bf16* qr = q + (size_t(t) * H + hg * 16 + (lane & 15)) * D + dh * 256 + (lane >> 4) * 8;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) qf[ks] = *reinterpret_cast<const bf16x8_t*>(qr + ks * 16);
  }
  for (int b = b0; b < b1; ++b) {
    const int e0 = b * KB;
    if (tid < KB) rows[tid] = kr.row(e0 + tid, t);
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

__device__ __forceinline__ void attn_pv_quarter(const int32_t* __restrict__ pos, const AttnKv& kv,
                                                const float* __restrict__ sink, bf16* __restrict__ o, int t, int dq,
                                                const float* __restrict__ S, int ldS) {
#if defined(UH_GFX12)
  __shared__ alignas(16) uint16_t sVt[128][KB + 8];  // [dim of the quarter][key]
  __shared__ alignas(16) uint16_t sP[H][KB + 8];
  __shared__ float sF[H], sM[H], sL[H];
  __shared__ const bf16* rows[KB];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int hg = wave >> 1, part = wave & 1;  // heads 16 hg .., dims dq * 128 + part * 64 ..
  const KeyRange kr(kv, pos, t);
  float8_t acc[4];
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[n][e] = 0.0f;
  if (tid < H) {
    sM[tid] = -INFINITY;
    sL[tid] = 0.0f;
  }
  for (int e0 = 0; e0 < kr.total; e0 += KB) {
    if (tid < KB) rows[tid] = kr.row(e0 + tid, t);
    __syncthreads();
    {  // V^T of the quarter: thread -> key tid % 16, 8 dims at (tid / 16) * 8
      const int key = tid & 15, d0 = (tid >> 4) * 8;
      const bf16* r = rows[key];
      const uint4 v = r ? *reinterpret_cast<const uint4*>(r + dq * 128 + d0) : uint4{0, 0, 0, 0};
      const uint16_t* hv = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
      for (int j = 0; j < 8; ++j) sVt[d0 + j][key] = hv[j];
    }
    {  // online softmax: the fused kernel's code, scores from S
      const int h = tid >> 2, k0 = (tid & 3) * 4;
      float sv[4], bm = -INFINITY;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sv[j] = S[size_t(h) * ldS + e0 + k0 + j];
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
    {
      const bf16x8_t pf = *reinterpret_cast<const bf16x8_t*>(&sP[hg * 16 + (lane & 15)][(lane >> 4) * 8]);
      float fr[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) fr[e] = sF[hg * 16 + (lane >> 4) * 8 + e];
#pragma unroll
      for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[n][e] *= fr[e];
        const bf16x8_t vf =
            *reinterpret_cast<const bf16x8_t*>(&sVt[part * 64 + n * 16 + (lane & 15)][(lane >> 4) * 8]);
        acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(pf, vf, acc[n]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int h = hg * 16 + (lane >> 4) * 8 + e;
    const float denom = sL[h] + expf(sink[h] - sM[h]);
    bf16* orow = o + (size_t(t) * H + h) * D + dq * 128 + part * 64 + (lane & 15);
#pragma unroll
    for (int n = 0; n < 4; ++n) orow[n * 16] = f32_to_bf16(acc[n][e] / denom);
  }
#endif
}

__global__ __launch_bounds__(NT) void attn_scores_kernel(const bf16* __restrict__ q, const int32_t* __restrict__ pos,
                                                         AttnKv kv, float scale, float* __restrict__ S, int ldS) {
  attn_scores(q, pos, kv, scale, blockIdx.x, blockIdx.y, S + size_t(blockIdx.x) * H * ldS, ldS);
}
__global__ __launch_bounds__(NT) void attn_pv_kernel(const int32_t* __restrict__ pos, AttnKv kv,
                                                     const float* __restrict__ sink, bf16* __restrict__ o,
                                                     const float* __restrict__ S, int ldS) {
  attn_pv_quarter(pos, kv, sink, o, blockIdx.x, blockIdx.y, S + size_t(blockIdx.x) * H * ldS, ldS);
}

// The batch's chunk of block x (as sparse_attn_wmma_batch_kernel): its AttnKv, first row and the local query.
__device__ __forceinline__ AttnKv chunk_kv(const AttnBatch& b, int& t0, int& tl) {
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
  return kv;
}
__global__ __launch_bounds__(NT) void attn_scores_batch_kernel(const bf16* __restrict__ q,
                                                               const int32_t* __restrict__ pos, AttnBatch b,
                                                               float scale, float* __restrict__ S, int ldS) {
  int t0 = 0, tl = 0;
  const AttnKv kv = chunk_kv(b, t0, tl);
  attn_scores(q + size_t(t0) * H * D, pos + t0, kv, scale, tl, blockIdx.y, S + size_t(blockIdx.x) * H * ldS, ldS);
}
__global__ __launch_bounds__(NT) void attn_pv_batch_kernel(const int32_t* __restrict__ pos, AttnBatch b,
                                                           const float* __restrict__ sink, bf16* __restrict__ o,
                                                           const float* __restrict__ S, int ldS) {
  int t0 = 0, tl = 0;
  const AttnKv kv = chunk_kv(b, t0, tl);
  attn_pv_quarter(pos + t0, kv, sink, o + size_t(t0) * H * D, tl, blockIdx.y, S + size_t(blockIdx.x) * H * ldS, ldS);
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

bool sparse_attn_wmma_split(const bf16* q, const int32_t* pos, AttnKv kv, const float* sink, bf16* o, int T, float scale,
                            int max_total, float* S, size_t s_floats, hipStream_t st) {
  if (T <= 0) return true;
  const int ldS = (max_total + KB - 1) / KB * KB;
  if (kv.dspark_slots >= 0 || size_t(T) * H * ldS > s_floats) return false;
  const int groups = (ldS / KB + SB - 1) / SB;
  attn_scores_kernel<<<dim3(T, groups), NT, 0, st>>>(q, pos, kv, scale, S, ldS);
  attn_pv_kernel<<<dim3(T, 4), NT, 0, st>>>(pos, kv, sink, o, S, ldS);
  UH_CHECK_LAUNCH();
  return true;
}

bool sparse_attn_wmma_batch_split(const bf16* q, const int32_t* pos, const AttnBatch& b, const float* sink, bf16* o,
                                  float scale, int max_total, float* S, size_t s_floats, hipStream_t st) {
  int n = 0;
  for (int i = 0; i < b.nchunks; ++i) n += b.c[i].n;
  if (n == 0) return true;
  const int ldS = (max_total + KB - 1) / KB * KB;
  if (size_t(n) * H * ldS > s_floats) return false;
  const int groups = (ldS / KB + SB - 1) / SB;
  attn_scores_batch_kernel<<<dim3(n, groups), NT, 0, st>>>(q, pos, b, scale, S, ldS);
  attn_pv_batch_kernel<<<dim3(n, 4), NT, 0, st>>>(pos, b, sink, o, S, ldS);
  UH_CHECK_LAUNCH();
  return true;
}

}  // namespace unholy::kernels

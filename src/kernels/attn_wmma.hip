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
  __shared__ const bf16* rows[KB];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int hg = wave >> 1, dh = wave & 1;
  const int p_t = pos[t];
  const int visible = kv.ratio ? (p_t + 1) / kv.ratio : 0;
  const int ncomp = kv.ratio == 0 ? 0 : (kv.ncomp > 0 ? kv.ncomp : visible);
  const int W = kv.vis ? 512 : 128;  // window entries: positions [lo, hi] from lo up (image spans: 512)
  const int lo = kv.vis ? kv.vis[2 * t] : max(0, p_t - 127), hi = kv.vis ? kv.vis[2 * t + 1] : p_t;
  const int total = W + ncomp;

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

  for (int e0 = 0; e0 < total; e0 += KB) {
    if (tid < KB) {
      const int e = e0 + tid;
      const bf16* r = nullptr;
      if (e < W) {
        const int wp = lo + e;
        if (wp <= hi) r = wp >= kv.s0 ? kv.chunk_kv + size_t(wp - kv.s0) * D : kv.ring + size_t(wp % 128) * D;
      } else if (e < total) {
        const int ci = e - W;
        const int row = kv.ncomp > 0 ? kv.comp_idx[size_t(t) * kv.ncomp + ci] : ci;
        if (row >= 0 && row < visible) r = paged_row(kv.comp, row);
      }
      rows[tid] = r;
    }
    __syncthreads();
    // Stage 16 rows x 512 dims: thread -> key tid/16, 32 dims at (tid%16)*32.
    {
      const int key = tid >> 4, d0 = (tid & 15) * 32;
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

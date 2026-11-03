// Skinny GEMMs (M <= 32 rows: decode, small batches): one wave per output column, K split across the
// 32 lanes in 16-element chunks, a wave reduction at the end. Bandwidth-bound on the weights, which
// is the right bound at small M (the 64-row WMMA tiles leave most of the GPU idle there).
//
// Numerics follow the WMMA/reference kernels: fp32 accumulation; the W8A8 path scales each 16-wide
// chunk's exact partial sum by its 128-block scale product (the reference scales whole 128-blocks; the
// same products, summed in a different fp32 order).
#include <algorithm>
#include <cstdlib>
#include <type_traits>

#include "kernels/dev.hip.hpp"
#include "kernels/gemv.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

constexpr int WAVES = 8, NT = WAVES * 32, MAXM = 32;  // MAXM: rows supported (kernels templated on R <= MAXM)

__device__ __forceinline__ void load16_fp8(const uint8_t* p, float* v) {
  const uint4 q = *reinterpret_cast<const uint4*>(p);
  e4m3x4_to_f32(q.x, v);
  e4m3x4_to_f32(q.y, v + 4);
  e4m3x4_to_f32(q.z, v + 8);
  e4m3x4_to_f32(q.w, v + 12);
}
__device__ __forceinline__ void load16_bf16(const bf16* p, float* v) {
  const uint4 q0 = *reinterpret_cast<const uint4*>(p), q1 = *reinterpret_cast<const uint4*>(p + 8);
  const uint32_t w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[2 * i] = __uint_as_float(w[i] << 16);
    v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

template <typename TOut>
__device__ __forceinline__ void store(TOut* c, size_t i, float v) {
  if constexpr (sizeof(TOut) == 2) c[i] = f32_to_bf16(v);
  else c[i] = v;
}

// A wave computes C columns (n0 .. n0 + C - 1; C = 1 by default): per column, the lanes' K partition, the
// 16-element fma chains, the scaling and the wave reduction do not depend on C. Rows are unrolled up to
// MAXM with uniform guards (m < M). fp8 decodes use gfx12's conversion instructions (dev::e4m3x4_to_f32):
// the bit decode made the W8A8 GEMV ALU-bound (~380 GB/s at one row, ~85 GB/s at eight).

__device__ __forceinline__ void cvt16_fp8(const uint4 q, float* v) {
  e4m3x4_to_f32(q.x, v);
  e4m3x4_to_f32(q.y, v + 4);
  e4m3x4_to_f32(q.z, v + 8);
  e4m3x4_to_f32(q.w, v + 12);
}

// W8A8: the wave's columns n0 .. n0 + C - 1 (< N).
// kLds (M >= 2; every wave of the block calls it, valid: the wave has columns): the activation rows of each
// 512-input step and their scales come from LDS, staged once per block a step ahead (each thread's share of step
// s + 1 loads while step s is computed, then one barrier), instead of every wave loading them: at 6 rows those
// loads were ~6x the weights' bytes. The same values and arithmetic.
template <int C, int R, bool kLds = false>
__device__ __forceinline__ void w8a8_cols(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw,
                                          bf16* c, int ldc, int M, int N, int K, int n0, int lane,
                                          const HcPostArgs& hp, bool valid = true) {
  const int nc = valid ? min(C, N - n0) : 0, nkb = K / 128;
  constexpr int PF = (R * 32 + NT - 1) / NT;  // 16-byte pieces of a step's rows per thread
  uint8_t(*sA)[R][512] = nullptr;
  float(*sS)[R][4] = nullptr;
  uint4 pa[PF] = {};
  float ps = 0.0f;
  const auto prefetch = [&](int k0) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      const int piece = int(threadIdx.x) + i * NT, m = piece >> 5;
      if (m < M) pa[i] = *reinterpret_cast<const uint4*>(a + size_t(m) * K + k0 + (piece & 31) * 16);
    }
    if (int(threadIdx.x) < M * 4) ps = e8m0_to_f32(sa[size_t(threadIdx.x >> 2) * nkb + k0 / 128 + (threadIdx.x & 3)]);
  };
  const auto stage = [&](int buf) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      const int piece = int(threadIdx.x) + i * NT, m = piece >> 5;
      if (m < M) *reinterpret_cast<uint4*>(&sA[buf][m][(piece & 31) * 16]) = pa[i];
    }
    if (int(threadIdx.x) < M * 4) sS[buf][threadIdx.x >> 2][threadIdx.x & 3] = ps;
  };
  if constexpr (kLds) {
    __shared__ alignas(16) uint8_t lds_a[2][R][512];
    __shared__ float lds_s[2][R][4];
    sA = lds_a;
    sS = lds_s;
    prefetch(0);
    stage(0);
    lds_barrier();
  }
  // hp.x (C == 1, M <= 8, hc == 4): lane 4 m + j holds row m's stream j at this column, its post weight and comb
  // column, loaded before the K loop (in the stores they would add their latency to every wave's tail).
  const int hm = lane >> 2, hj = lane & 3;
  float hr = 0.0f, hpj = 0.0f, hcj[4] = {};
  if (valid && hp.x && hm < M) {
    hr = bf16_to_f32(hp.x[(size_t(hm) * 4 + hj) * N + n0]);
    hpj = hp.post[hm * 4 + hj];
#pragma unroll
    for (int i = 0; i < 4; ++i) hcj[i] = hp.comb[hm * 16 + i * 4 + hj];
  }
  float acc[C][R] = {};
  for (int k0 = 0, step = 0; k0 < K; k0 += 512, ++step) {
    const int k = k0 + lane * 16, kb = k / 128, buf = step & 1;
    float wv[C][16], ws[C];
#pragma unroll
    for (int j = 0; j < C; ++j)
      if (j < nc) {
        load16_fp8(w + size_t(n0 + j) * K + k, wv[j]);
        ws[j] = e8m0_to_f32(sw[size_t((n0 + j) / 128) * nkb + kb]);
      }
    if constexpr (kLds) {
      if (k0 + 512 < K) prefetch(k0 + 512);
    }
#pragma unroll
    for (int m = 0; m < R; ++m) {
      if (m >= M || nc == 0) break;
      float av[16], sm;
      if constexpr (kLds) {
        cvt16_fp8(*reinterpret_cast<const uint4*>(&sA[buf][m][lane * 16]), av);
        sm = sS[buf][m][lane >> 3];
      } else {
        load16_fp8(a + size_t(m) * K + k, av);
        sm = e8m0_to_f32(sa[size_t(m) * nkb + kb]);
      }
#pragma unroll
      for (int j = 0; j < C; ++j) {
        if (j >= nc) break;
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) part = fmaf(av[i], wv[j][i], part);
        acc[j][m] += part * (sm * ws[j]);
      }
    }
    if constexpr (kLds) {
      if (k0 + 512 < K) {
        stage(buf ^ 1);
        lds_barrier();
      }
    }
  }
  if (nc == 0) return;
#pragma unroll
  for (int j = 0; j < C; ++j)
#pragma unroll
    for (int m = 0; m < R; ++m) {
      if (j >= nc || m >= M) break;
      const float v = warp_sum(acc[j][m]);
      const bf16 o = f32_to_bf16(v);
      if (lane == 0) c[size_t(m) * ldc + n0 + j] = o;
      if (hp.x) {  // hc_post_column's arithmetic, stream hj of row m in lane 4 m + hj
        const float r[4] = {quad_lane<0>(hr), quad_lane<1>(hr), quad_lane<2>(hr), quad_lane<3>(hr)};
        if (hm == m) hp.x[(size_t(m) * 4 + hj) * N + n0] = f32_to_bf16(hc_post_value(bf16_to_f32(o), r, hcj, hpj, 4));
      }
    }
}
// act_quant_fp8_kernel's arithmetic on one 128-value block held by a wave (lane l: values 4 l .. 4 l + 3): codes into
// q[0 .. 128), the scale exponent into *qs.
__device__ __forceinline__ void quant128_wave(const float v[4], uint8_t* q, uint8_t* qs, int lane) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(v[i]));
  amax = fmaxf(warp_max(amax), 1e-4f);
  const uint8_t se = pow2_ceil_e8m0(amax * (1.0f / 448.0f));
  const float sc = e8m0_to_f32(se);
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) packed |= uint32_t(f32_to_e4m3(fminf(fmaxf(v[i] / sc, -448.0f), 448.0f))) << (8 * i);
  *reinterpret_cast<uint32_t*>(q + 4 * lane) = packed;
  if (lane == 0) *qs = se;
}
// Four bf16 values another block of the launch stored (after its fence), as floats.
__device__ __forceinline__ void load4_peer(const bf16* p, float v[4]) {
  const uint64_t r = __hip_atomic_load(reinterpret_cast<const uint64_t*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = bf16_to_f32(uint16_t(r >> (16 * i)));
}
// The block's turn on a segment counter: true for the target-th block to arrive (it then resets the counter).
__device__ __forceinline__ bool last_block(uint32_t* cnt, uint32_t target) {
  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(cnt, 1u) == target - 1;
  __syncthreads();
  if (!last) return false;
  __threadfence();
  if (threadIdx.x == 0) *cnt = 0;
  return true;
}

template <int C, int R>
__global__ __launch_bounds__(NT) void gemv_w8a8_kernel(const uint8_t* a, const uint8_t* sa, const uint8_t* w,
                                                       const uint8_t* sw, bf16* c, int ldc, int M, int N, int K,
                                                       HcPostArgs hp) {
  const int n0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * C;
  if (M >= 2) {
    if (int(blockIdx.x) * WAVES * C < N)  // blocks with a column: all their waves take the staging's barriers
      w8a8_cols<C, R, true>(a, sa, w, sw, c, ldc, M, N, K, n0, threadIdx.x & 31, hp, n0 < N);
  } else if (n0 < N) {
    w8a8_cols<C, R>(a, sa, w, sw, c, ldc, M, N, K, n0, threadIdx.x & 31, hp);
  }
}
// gemv_w8a8_parts_kernel on [gate; up] (two parts of width I), then each segment's last block (of 32: 16 per part):
// swiglu_kernel's arithmetic and the quantization of its products (a wave per row).
template <int R>
__global__ __launch_bounds__(NT) void gemv_w8a8_glu_kernel(const uint8_t* a, const uint8_t* sa, GemvParts ps, int M,
                                                           int K, float limit, GemvQuant gq) {
  const int I = ps.p[0].N;
  int n0 = blockIdx.x * WAVES + (threadIdx.x >> 5);
  const int part = n0 >= I;
  n0 -= part * I;
  if (n0 >= I) return;  // pad blocks (whole blocks: I % 8 == 0)
  const GemvPart& p = ps.p[part];
  if (M >= 2)
    w8a8_cols<1, R, true>(a, sa, static_cast<const uint8_t*>(p.w), p.sw, static_cast<bf16*>(p.c), p.ldc, M, I, K, n0,
                          threadIdx.x & 31, HcPostArgs{});
  else
    w8a8_cols<1, R>(a, sa, static_cast<const uint8_t*>(p.w), p.sw, static_cast<bf16*>(p.c), p.ldc, M, I, K, n0,
                    threadIdx.x & 31, HcPostArgs{});
  const int seg = n0 / 128, lane = threadIdx.x & 31;
  if (!last_block(gq.counters + seg, 32)) return;
  const bf16* g = static_cast<const bf16*>(ps.p[0].c);
  const bf16* u = static_cast<const bf16*>(ps.p[1].c);
  for (int m = threadIdx.x >> 5; m < M; m += WAVES) {
    const int k0 = seg * 128 + 4 * lane;
    float gv[4], uv[4], v[4];
    load4_peer(g + size_t(m) * ps.p[0].ldc + k0, gv);
    load4_peer(u + size_t(m) * ps.p[1].ldc + k0, uv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float gi = gv[i], ui = uv[i];
      if (limit > 0.0f) {
        ui = fminf(fmaxf(ui, -limit), limit);
        gi = fminf(gi, limit);
      }
      v[i] = round_bf16((gi / (1.0f + expf(-gi))) * ui);
    }
    quant128_wave(v, gq.q + size_t(m) * I + seg * 128, gq.qs + size_t(m) * (I / 128) + seg, lane);
  }
}

// Several column ranges (parts with their own weights, scales and outputs) on one input: wave -> (part,
// column), then the single GEMV's code.
template <int R>
__global__ __launch_bounds__(NT) void gemv_w8a8_parts_kernel(const uint8_t* a, const uint8_t* sa, GemvParts ps, int M,
                                                             int K) {
  int n0 = blockIdx.x * WAVES + (threadIdx.x >> 5), i = 0;
  while (i + 1 < ps.n && n0 >= ps.p[i].N) n0 -= ps.p[i++].N;
  const GemvPart& p = ps.p[i];
  if (M >= 2) {
    int cols = 0;
    for (int q = 0; q < ps.n; ++q) cols += ps.p[q].N;
    if (int(blockIdx.x) * WAVES < cols)
      w8a8_cols<1, R, true>(a, sa, static_cast<const uint8_t*>(p.w), p.sw, static_cast<bf16*>(p.c), p.ldc, M, p.N, K,
                            n0, threadIdx.x & 31, HcPostArgs{}, n0 < p.N);
  } else if (n0 < p.N) {
    w8a8_cols<1, R>(a, sa, static_cast<const uint8_t*>(p.w), p.sw, static_cast<bf16*>(p.c), p.ldc, M, p.N, K, n0,
                    threadIdx.x & 31, HcPostArgs{});
  }
}

// bf16 weights x bf16 activations: the wave's columns n0 .. n0 + C - 1 (< N).
// scale != 0 (bf16 outputs): scale_bf16 on each output. ring (fp32 outputs, C == 1, part: its index): rg's stores,
// row m's position and ape value loaded into lane m before the K loop.
// bf16 activation rows for 512-input steps in LDS, staged a step ahead (bf16_cols' kLds path, as w8a8_cols'): row m
// at a + m * lda; each thread's 16-byte pieces of step s + 1 load while step s is computed.
template <int R>
struct Bf16Stage {
  static constexpr int PF = (R * 64 + NT - 1) / NT;
  uint16_t (*s)[R][512];
  uint4 pa[PF];
  __device__ __forceinline__ void prefetch(const bf16* a, size_t lda, int M, int k0) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      const int piece = int(threadIdx.x) + i * NT, m = piece >> 6;
      if (m < M) pa[i] = *reinterpret_cast<const uint4*>(a + size_t(m) * lda + k0 + (piece & 63) * 8);
    }
  }
  __device__ __forceinline__ void stage(int buf, int M) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      const int piece = int(threadIdx.x) + i * NT, m = piece >> 6;
      if (m < M) *reinterpret_cast<uint4*>(&s[buf][m][(piece & 63) * 8]) = pa[i];
    }
  }
  __device__ __forceinline__ void load(int buf, int m, int lane, float* v) const {
    const uint4 q0 = *reinterpret_cast<const uint4*>(&s[buf][m][lane * 16]);
    const uint4 q1 = *reinterpret_cast<const uint4*>(&s[buf][m][lane * 16 + 8]);
    const uint32_t w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[2 * i] = __uint_as_float(w[i] << 16);
      v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  }
};

// kLds (M >= 2, R <= 16; every wave of the block calls it, valid: the wave has columns): the activations through
// Bf16Stage. The same values and arithmetic.
template <int C, int R, typename TOut, bool kLds = false>
__device__ __forceinline__ void bf16_cols(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M, int N, int K,
                                          int n0, int lane, float scale = 0.0f, bool ring = false, GemvRing rg = {},
                                          int part = 0, bool valid = true) {
  const int nc = valid ? min(C, N - n0) : 0;
  Bf16Stage<R> sg;
  if constexpr (kLds) {
    __shared__ alignas(16) uint16_t lds_a[2][R][512];
    sg.s = lds_a;
    sg.prefetch(a, lda, M, 0);
    sg.stage(0, M);
    lds_barrier();
  }
  int rpos = 0;
  float rape = 0.0f;
  if (valid && ring && lane < M) {
    rpos = rg.pos[lane];
    if (part == 1) rape = rg.ape[size_t(rpos % rg.ratio) * N + n0];
  }
  float acc[C][R] = {};
  for (int k0 = 0, step = 0; k0 < K; k0 += 512, ++step) {
    const int k = k0 + lane * 16, buf = step & 1;
    float wv[C][16];
#pragma unroll
    for (int j = 0; j < C; ++j)
      if (j < nc) load16_bf16(w + size_t(n0 + j) * K + k, wv[j]);
    if constexpr (kLds) {
      if (k0 + 512 < K) sg.prefetch(a, lda, M, k0 + 512);
    }
#pragma unroll
    for (int m = 0; m < R; ++m) {
      if (m >= M || nc == 0) break;
      float av[16];
      if constexpr (kLds) sg.load(buf, m, lane, av);
      else load16_bf16(a + size_t(m) * lda + k, av);
#pragma unroll
      for (int j = 0; j < C; ++j) {
        if (j >= nc) break;
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[j][m] = fmaf(av[i], wv[j][i], acc[j][m]);
      }
    }
    if constexpr (kLds) {
      if (k0 + 512 < K) {
        sg.stage(buf ^ 1, M);
        lds_barrier();
      }
    }
  }
  if (nc == 0) return;
#pragma unroll
  for (int j = 0; j < C; ++j)
#pragma unroll
    for (int m = 0; m < R; ++m) {
      if (j >= nc || m >= M) break;
      float v = warp_sum(acc[j][m]);
      int p = 0;
      if (ring) {
        p = __builtin_amdgcn_readlane(rpos, m);
        if (part == 1) v += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rape), m));
      }
      if (lane != 0) continue;
      if constexpr (sizeof(TOut) == 2) {
        if (scale != 0.0f) v = bf16_to_f32(f32_to_bf16(v)) * scale;
      } else if (ring && rg.slot[m] >= 0) {
        (part ? rg.sc : rg.kv)[(size_t(rg.slot[m]) * rg.ring + p % rg.ring) * N + n0 + j] = v;
      }
      store(c, size_t(m) * ldc + n0 + j, v);
    }
}
template <int C, int R, typename TOut>
__global__ __launch_bounds__(NT) void gemv_bf16_kernel(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M,
                                                       int N, int K, float scale) {
  const int n0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * C;
  if constexpr (R <= 16) {
    if (M >= 2 && N <= 65536) {  // not the LM head: its waves' activation reads hit in L1 already (5% slower staged)
      if (int(blockIdx.x) * WAVES * C < N)  // blocks with a column: all their waves take the staging's barriers
        bf16_cols<C, R, TOut, true>(a, lda, w, c, ldc, M, N, K, n0, threadIdx.x & 31, scale, false, {}, 0, n0 < N);
      return;
    }
  }
  if (n0 < N) bf16_cols<C, R, TOut>(a, lda, w, c, ldc, M, N, K, n0, threadIdx.x & 31, scale);
}
// K < 512 (K % 64 == 0; KDA's low-rank gate projections, K = 128): a wave per 8 columns, 4 lanes per column, each
// lane K / 4 consecutive values of it (the per-512 layout above would leave most lanes idle).
template <int R, typename TOut>
__device__ __forceinline__ void bf16_smallk_cols(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M, int N,
                                                 int K, int block) {
  const int lane = threadIdx.x & 31, n = (block * WAVES + (threadIdx.x >> 5)) * 8 + (lane >> 2);
  const int kq = K / 4, k0 = (lane & 3) * kq;
  float acc[R] = {};
  if (n < N)
    for (int k = k0; k < k0 + kq; k += 16) {
      float wv[16];
      load16_bf16(w + size_t(n) * K + k, wv);
#pragma unroll
      for (int m = 0; m < R; ++m) {
        if (m >= M) break;
        float av[16];
        load16_bf16(a + size_t(m) * lda + k, av);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[m] = fmaf(av[i], wv[i], acc[m]);
      }
    }
#pragma unroll
  for (int m = 0; m < R; ++m) {
    if (m >= M) break;
    float v = acc[m];
    v += xor_lane<1>(v);
    v += xor_lane<2>(v);
    if ((lane & 3) == 0 && n < N) store(c, size_t(m) * ldc + n, v);
  }
}
template <int R, typename TOut>
__global__ __launch_bounds__(NT) void gemv_bf16_smallk_kernel(const bf16* a, int lda, const bf16* w, TOut* c, int ldc,
                                                              int M, int N, int K) {
  bf16_smallk_cols<R, TOut>(a, lda, w, c, ldc, M, N, K, blockIdx.x);
}
// Blocks [0, nb0): the first GEMV, the rest the second.
template <int R>
__global__ __launch_bounds__(NT) void gemv_bf16_smallk2_kernel(const bf16* a0, int lda0, const bf16* w0, bf16* c0,
                                                               int ldc0, int N0, const bf16* a1, int lda1,
                                                               const bf16* w1, bf16* c1, int ldc1, int N1, int M, int K,
                                                               int nb0) {
  if (int(blockIdx.x) < nb0) bf16_smallk_cols<R, bf16>(a0, lda0, w0, c0, ldc0, M, N0, K, blockIdx.x);
  else bf16_smallk_cols<R, bf16>(a1, lda1, w1, c1, ldc1, M, N1, K, blockIdx.x - nb0);
}
template <int R, typename TOut>
__global__ __launch_bounds__(NT) void gemv_bf16_parts_kernel(const bf16* a, int lda, GemvParts ps, int M, int K,
                                                             GemvRing rg) {
  int n0 = blockIdx.x * WAVES + (threadIdx.x >> 5), i = 0;
  while (i + 1 < ps.n && n0 >= ps.p[i].N) n0 -= ps.p[i++].N;
  const GemvPart& p = ps.p[i];
  if constexpr (R <= 16) {
    if (M >= 2) {
      int cols = 0;
      for (int q = 0; q < ps.n; ++q) cols += ps.p[q].N;
      if (int(blockIdx.x) * WAVES < cols)
        bf16_cols<1, R, TOut, true>(a, lda, static_cast<const bf16*>(p.w), static_cast<TOut*>(p.c), p.ldc, M, p.N, K,
                                    n0, threadIdx.x & 31, 0.0f, rg.ape != nullptr, rg, i, n0 < p.N);
      return;
    }
  }
  if (n0 < p.N)
    bf16_cols<1, R, TOut>(a, lda, static_cast<const bf16*>(p.w), static_cast<TOut*>(p.c), p.ldc, M, p.N, K, n0,
                          threadIdx.x & 31, 0.0f, rg.ape != nullptr, rg, i);
}

// Grouped "mgk,grk->mgr": fp8 weights x 128x128 scales (exact in bf16), bf16 activations. A wave's C
// columns lie in one group (R % C == 0).
// gq.q: then each 128-column segment's last block (C == 1: 16 blocks of 8 columns) quantizes its rows.
template <int C, int RM>
__global__ __launch_bounds__(NT) void gemv_grouped_fp8w_kernel(const bf16* a, const uint8_t* w, const uint8_t* sw,
                                                               bf16* c, int M, int G, int R, int K, GemvQuant gq) {
  const int col0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * C, lane = threadIdx.x & 31;  // col over G*R
  if (col0 >= G * R) return;
  const int g = col0 / R, r0 = col0 % R;
  float acc[C][RM] = {};
  for (int k = lane * 16; k < K; k += 512) {
    float wv[C][16];
#pragma unroll
    for (int j = 0; j < C; ++j) {
      load16_fp8(w + size_t(col0 + j) * K + k, wv[j]);
      const float s = e8m0_to_f32(sw[size_t((col0 + j) / 128) * (K / 128) + k / 128]);
#pragma unroll
      for (int i = 0; i < 16; ++i) wv[j][i] *= s;  // exact (power of two)
    }
#pragma unroll
    for (int m = 0; m < RM; ++m) {
      if (m >= M) break;
      float av[16];
      load16_bf16(a + (size_t(m) * G + g) * K + k, av);
#pragma unroll
      for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[j][m] = fmaf(av[i], wv[j][i], acc[j][m]);
    }
  }
#pragma unroll
  for (int j = 0; j < C; ++j)
#pragma unroll
    for (int m = 0; m < RM; ++m) {
      if (m >= M) break;
      const float v = warp_sum(acc[j][m]);
      if (lane == 0) c[(size_t(m) * G + g) * R + r0 + j] = f32_to_bf16(v);
    }
  if (!gq.q) return;
  const int N = G * R, seg = col0 / 128;
  if (!last_block(gq.counters + seg, 128 / (WAVES * C))) return;
  for (int m = threadIdx.x >> 5; m < M; m += WAVES) {
    float v[4];
    load4_peer(c + size_t(m) * N + seg * 128 + 4 * lane, v);
    quant128_wave(v, gq.q + size_t(m) * N + seg * 128, gq.qs + size_t(m) * (N / 128) + seg, lane);
  }
}

int g_force_cols = 0;
// On gfx1201 a kernel whose grid fills every wave slot of the card exactly (2048 waves: 64 CUs at 16 per SIMD, e.g.
// 253-256 blocks of 8 waves; also 509-512) runs its blocks in ~6 us but takes ~22 us to complete; with LDS or
// registers limiting the blocks per CU it does not, and a kernel much longer than that hides most of it. For these
// 8-wave blocks one idle block per shader engine more (the GEMVs skip columns past N) avoids it: KDA's two
// 8192-column small-K GEMVs (256 blocks) 22 -> 9 us in decode, DSV4's shared w2 (512) 37 -> 21 us.
int pad_grid(int blocks) { return blocks % 256 == 0 ? blocks + 4 : blocks; }

// Columns per wave: 1. With the weights streamed from DRAM (unholy-gemv-test) more columns per wave only
// cost occupancy; the template stays for tuning (gemv_force_cols).
int cols_per_wave(int) { return g_force_cols ? g_force_cols : 1; }
int blocks_for(int cols, int C) { return pad_grid((cols + WAVES * C - 1) / (WAVES * C)); }

// Calls f(std::integral_constant<int, R>) with the smallest row capacity R in {8, 16, 32} >= M.
template <typename F>
void by_rows(int M, F&& f) {
  if (M <= 8) f(std::integral_constant<int, 8>{});
  else if (M <= 16) f(std::integral_constant<int, 16>{});
  else f(std::integral_constant<int, 32>{});
}

}  // namespace

int gemv_max_rows() {
  static const int v = [] {
    const char* e = std::getenv("UNHOLY_GEMV_ROWS");
    return e ? std::clamp(std::atoi(e), 1, int(kGemvRowsCap)) : 8;
  }();
  return v;
}

void gemv_force_cols(int c) { g_force_cols = c == 1 || c == 2 || c == 4 ? c : 0; }

void gemv_w8a8(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc, int M,
               int N, int K, hipStream_t st, const HcPostArgs& hp) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_w8a8: M > 32 or K % 512");
  if (hp.x && (M > 8 || hp.hc != 4 || cols_per_wave(N) != 1)) throw std::runtime_error("gemv_w8a8: post-mix shape");
  const auto go = [&](auto r) {
    constexpr int R = decltype(r)::value;
    switch (cols_per_wave(N)) {
      case 4: gemv_w8a8_kernel<4, R><<<blocks_for(N, 4), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K, hp); break;
      case 2: gemv_w8a8_kernel<2, R><<<blocks_for(N, 2), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K, hp); break;
      default: gemv_w8a8_kernel<1, R><<<blocks_for(N, 1), NT, 0, st>>>(a, sa, w, sw, c, ldc, M, N, K, hp);
    }
  };
  by_rows(M, go);
  UH_CHECK_LAUNCH();
}
template <typename TOut>
void gemv_bf16_any(const bf16* a, int lda, const bf16* w, TOut* c, int ldc, int M, int N, int K, hipStream_t st,
                   float scale = 0.0f) {
  if (M <= MAXM && K < 512 && K % 64 == 0) {
    if (scale != 0.0f) throw std::runtime_error("gemv_bf16: scale needs K >= 512");
    by_rows(M, [&](auto r) {
      gemv_bf16_smallk_kernel<decltype(r)::value, TOut>
          <<<pad_grid((N + WAVES * 8 - 1) / (WAVES * 8)), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K);
    });
    UH_CHECK_LAUNCH();
    return;
  }
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_bf16: M > 32, or K % 512 (K >= 512) or K % 64");
  const auto go = [&](auto r) {
    constexpr int R = decltype(r)::value;
    switch (cols_per_wave(N)) {
      case 4: gemv_bf16_kernel<4, R, TOut><<<blocks_for(N, 4), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K, scale); break;
      case 2: gemv_bf16_kernel<2, R, TOut><<<blocks_for(N, 2), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K, scale); break;
      default: gemv_bf16_kernel<1, R, TOut><<<blocks_for(N, 1), NT, 0, st>>>(a, lda, w, c, ldc, M, N, K, scale);
    }
  };
  by_rows(M, go);
  UH_CHECK_LAUNCH();
}
void gemv_bf16(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st,
               float scale) {
  gemv_bf16_any(a, lda, w, c, ldc, M, N, K, st, scale);
}
void gemv_bf16_f32out(const bf16* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K, hipStream_t st) {
  gemv_bf16_any(a, lda, w, c, ldc, M, N, K, st);
}
namespace {
GemvParts make_parts(const GemvPart* parts, int nparts, int& cols) {
  if (nparts < 1 || nparts > GemvParts::kMax) throw std::runtime_error("gemv parts: 1..3");
  GemvParts ps;
  ps.n = nparts;
  cols = 0;
  for (int i = 0; i < nparts; ++i) {
    ps.p[i] = parts[i];
    cols += parts[i].N;
  }
  return ps;
}
}  // namespace

void gemv_w8a8_parts(const uint8_t* a, const uint8_t* sa, const GemvPart* parts, int nparts, int M, int K,
                     hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_w8a8_parts: M > 32 or K % 512");
  int cols = 0;
  const GemvParts ps = make_parts(parts, nparts, cols);
  by_rows(M, [&](auto r) {
    gemv_w8a8_parts_kernel<decltype(r)::value><<<blocks_for(cols, 1), NT, 0, st>>>(a, sa, ps, M, K);
  });
  UH_CHECK_LAUNCH();
}
void gemv_w8a8_glu(const uint8_t* a, const uint8_t* sa, const GemvPart* parts, int M, int K, float limit,
                   const GemvQuant& gq, hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_w8a8_glu: M > 32 or K % 512");
  if (parts[0].N != parts[1].N || parts[0].N % 128) throw std::runtime_error("gemv_w8a8_glu: two parts of one width");
  int cols = 0;
  const GemvParts ps = make_parts(parts, 2, cols);
  by_rows(M, [&](auto r) {
    gemv_w8a8_glu_kernel<decltype(r)::value><<<blocks_for(cols, 1), NT, 0, st>>>(a, sa, ps, M, K, limit, gq);
  });
  UH_CHECK_LAUNCH();
}
void gemv_bf16_f32out_parts(const bf16* a, int lda, const GemvPart* parts, int nparts, int M, int K, hipStream_t st,
                            const GemvRing* ring) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_bf16_f32out_parts: M > 32 or K % 512");
  if (ring && (nparts != 2 || parts[0].N != parts[1].N)) throw std::runtime_error("gemv ring: two parts of one width");
  int cols = 0;
  const GemvParts ps = make_parts(parts, nparts, cols);
  const GemvRing rg = ring ? *ring : GemvRing{};
  by_rows(M, [&](auto r) {
    gemv_bf16_parts_kernel<decltype(r)::value, float><<<blocks_for(cols, 1), NT, 0, st>>>(a, lda, ps, M, K, rg);
  });
  UH_CHECK_LAUNCH();
}

void gemv_bf16_parts(const bf16* a, int lda, const GemvPart* parts, int nparts, int M, int K, hipStream_t st) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_bf16_parts: M > 32 or K % 512");
  int cols = 0;
  const GemvParts ps = make_parts(parts, nparts, cols);
  by_rows(M, [&](auto r) {
    gemv_bf16_parts_kernel<decltype(r)::value, bf16><<<blocks_for(cols, 1), NT, 0, st>>>(a, lda, ps, M, K, GemvRing{});
  });
  UH_CHECK_LAUNCH();
}

void gemv_bf16_smallk2(const bf16* a0, int lda0, const bf16* w0, bf16* c0, int ldc0, int N0, const bf16* a1, int lda1,
                       const bf16* w1, bf16* c1, int ldc1, int N1, int M, int K, hipStream_t st) {
  if (M > MAXM || K >= 512 || K % 64) throw std::runtime_error("gemv_bf16_smallk2: M > 32, K >= 512 or K % 64");
  const int nb0 = (N0 + WAVES * 8 - 1) / (WAVES * 8), nb1 = (N1 + WAVES * 8 - 1) / (WAVES * 8);
  by_rows(M, [&](auto r) {
    gemv_bf16_smallk2_kernel<decltype(r)::value><<<pad_grid(nb0 + nb1), NT, 0, st>>>(a0, lda0, w0, c0, ldc0, N0, a1,
                                                                                       lda1, w1, c1, ldc1, N1, M, K, nb0);
  });
  UH_CHECK_LAUNCH();
}

void gemv_grouped_fp8w(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                       hipStream_t st, const GemvQuant& gq) {
  if (M > MAXM || K % 512) throw std::runtime_error("gemv_grouped_fp8w: M > 32 or K % 512");
  int C = cols_per_wave(G * R);
  while (R % C) C /= 2;
  if (gq.q && (C != 1 || (G * R) % 128)) throw std::runtime_error("gemv_grouped_fp8w: quantization shape");
  const auto go = [&](auto r) {
    constexpr int RR = decltype(r)::value;
    switch (C) {
      case 4: gemv_grouped_fp8w_kernel<4, RR><<<blocks_for(G * R, 4), NT, 0, st>>>(a, w, sw, c, M, G, R, K, gq); break;
      case 2: gemv_grouped_fp8w_kernel<2, RR><<<blocks_for(G * R, 2), NT, 0, st>>>(a, w, sw, c, M, G, R, K, gq); break;
      default: gemv_grouped_fp8w_kernel<1, RR><<<blocks_for(G * R, 1), NT, 0, st>>>(a, w, sw, c, M, G, R, K, gq);
    }
  };
  by_rows(M, go);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

// Device-side MoE dispatch: everything the host used to do between the router and the expert GEMMs.
//
//   moe_count     sel [T, K] -> count [E]
//   moe_plan      count + resident mask -> pair offsets (resident experts first, then remote, each in
//                 ascending expert order), remote0, and the <= kExl3TileRows-row tile lists of both groups
//   moe_scatter   per token: its K picks sorted by expert id; each pick gets a pair row
//                 (off[e] + atomic fill): eidx (token), ewt (weight), row_expert, pidx[t][j]
//
// The row order inside an expert's block depends on atomic arrival order; every expert kernel is
// row-independent and the combine follows pidx (ascending expert per token), so results do not.
#include <algorithm>
#include <stdexcept>

#include "kernels/dev.hip.hpp"
#include "kernels/elem_fast.hpp"
#include "kernels/moe_dispatch.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

__global__ void moe_count_kernel(const int32_t* sel, int n, int32_t* count, uint32_t* hist) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&count[sel[i]], 1);
  if (hist) atomicAdd(&hist[sel[i]], 1u);
}

// One block of 256 threads, thread e for expert e (E <= 256): block-wide scans of the pair rows and tiles
// of the resident group and the remote group give the same offsets and tile order as walking the experts
// in order, resident ones first.
__global__ __launch_bounds__(256) void moe_plan_kernel(int32_t* count, const uint8_t* resident, int E, int P,
                                                       MoePlan* plan, Exl3Tile* local_tiles, Exl3Tile* remote_tiles,
                                                       int32_t* off, int32_t* fill, int max_tiles, int tile_rows) {
  __shared__ int sc[4][256];  // inclusive scans: local rows, remote rows, local tiles, remote tiles
  const int e = threadIdx.x;
  const bool in = e < E, loc = in && resident[e] != 0;
  const int c = in ? count[e] : 0, nt = (c + tile_rows - 1) / tile_rows;
  const int v[4] = {loc ? c : 0, in && !loc ? c : 0, loc ? nt : 0, in && !loc ? nt : 0};
#pragma unroll
  for (int k = 0; k < 4; ++k) sc[k][e] = v[k];
  __syncthreads();
  for (int d = 1; d < 256; d <<= 1) {
    int add[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) add[k] = e >= d ? sc[k][e - d] : 0;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; ++k) sc[k][e] += add[k];
    __syncthreads();
  }
  const int remote0 = sc[0][255];
  if (in) {
    const int row = loc ? sc[0][e] - v[0] : remote0 + sc[1][e] - v[1];
    const int tile = loc ? sc[2][e] - v[2] : sc[3][e] - v[3];
    Exl3Tile* dst = loc ? local_tiles : remote_tiles;
    off[e] = row;
    for (int i = 0; i < nt && tile + i < max_tiles; ++i)
      dst[tile + i] = Exl3Tile{e, row + i * tile_rows, min(tile_rows, c - i * tile_rows)};
    fill[e] = 0;
    count[e] = 0;  // for the next moe_count (saves a memset launch per layer)
  }
  if (e == 0) {
    plan->P = P;
    plan->remote0 = remote0;
    plan->n_local_tiles = min(sc[2][255], max_tiles);
    plan->n_remote_tiles = min(sc[3][255], max_tiles);
    plan->n_remote_tokens = 0;
  }
}

__global__ void moe_scatter_kernel(const int32_t* sel, const float* wt, int T, int K, const int32_t* off, int32_t* fill,
                                   int32_t* eidx, float* ewt, int32_t* row_expert, int32_t* pidx) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int es[8];
  float ws[8];
  for (int j = 0; j < K; ++j) {
    es[j] = sel[size_t(t) * K + j];
    ws[j] = wt[size_t(t) * K + j];
  }
  for (int a = 1; a < K; ++a)  // insertion sort by expert id
    for (int b = a; b > 0 && es[b - 1] > es[b]; --b) {
      const int te = es[b];
      es[b] = es[b - 1];
      es[b - 1] = te;
      const float tw = ws[b];
      ws[b] = ws[b - 1];
      ws[b - 1] = tw;
    }
  for (int j = 0; j < K; ++j) {
    const int e = es[j];
    const int p = off[e] + atomicAdd(&fill[e], 1);
    eidx[p] = t;
    ewt[p] = ws[j];
    row_expert[p] = e;
    pidx[size_t(t) * K + j] = p;
  }
}

// Rows [plan->remote0, plan->P) of src -> dst (same row indices). Grid sized for the maximum.
__global__ void copy_remote_rows_kernel(const uint16_t* src, uint16_t* dst, const MoePlan* plan, int dim) {
  const int r0 = plan->remote0, r1 = plan->P;
  const size_t n = size_t(r1 - r0) * dim / 8;  // uint4 units
  const uint4* s = reinterpret_cast<const uint4*>(src + size_t(r0) * dim);
  uint4* d = reinterpret_cast<uint4*>(dst + size_t(r0) * dim);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    d[i] = s[i];
  __threadfence_system();
}

__global__ void copy_remote_meta_kernel(const float* ewt, const int32_t* row_expert, float* w_dst, int32_t* e_dst,
                                        const MoePlan* plan) {
  const int r0 = plan->remote0, r1 = plan->P;
  for (int i = r0 + blockIdx.x * blockDim.x + threadIdx.x; i < r1; i += gridDim.x * blockDim.x) {
    w_dst[i] = ewt[i];
    e_dst[i] = row_expert[i];
  }
  __threadfence_system();
}

// Everything the remote side needs for one layer, written by one kernel (so it is ordered with the
// stream's flag packets like any kernel): the plan, the remote tiles, and the routing weight and expert
// of the remote rows [remote0, P).
__global__ void stage_remote_kernel(const MoePlan* plan, const Exl3Tile* rtiles, const float* ewt,
                                    const int32_t* row_expert, MoePlan* plan_dst, Exl3Tile* tiles_dst, float* w_dst,
                                    int32_t* e_dst) {
  const MoePlan p = *plan;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, n = gridDim.x * blockDim.x;
  if (tid == 0) *plan_dst = p;
  for (int i = tid; i < p.n_remote_tiles; i += n) tiles_dst[i] = rtiles[i];
  for (int i = p.remote0 + tid; i < p.P; i += n) {
    w_dst[i] = ewt[i];
    e_dst[i] = row_expert[i];
  }
  __threadfence_system();
}

// One block of 1024 threads: an exclusive scan of "has a remote pair" over the tokens, 1024 at a time.
__global__ __launch_bounds__(1024) void remote_tokens_kernel(const int32_t* pidx, const int32_t* eidx, int T, int K,
                                                             MoePlan* plan, int32_t* tok2u, int32_t* u2t,
                                                             int32_t* p2u, int32_t* u2p) {
  __shared__ int wsum[32];
  __shared__ int carry;
  const int r0 = plan->remote0, P = plan->P;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, nw = blockDim.x >> 5;
  if (tid == 0) carry = 0;
  __syncthreads();
  for (int base = 0; base < T; base += blockDim.x) {
    const int t = base + tid;
    int f = 0;
    if (t < T)
      for (int j = 0; j < K; ++j) f |= pidx[size_t(t) * K + j] >= r0;
    int v = f;  // inclusive scan within the wave
    for (int o = 1; o < 32; o <<= 1) {
      const int n = __shfl_up(v, o, 32);
      if (lane >= o) v += n;
    }
    if (lane == 31) wsum[wave] = v;
    __syncthreads();
    if (wave == 0) {
      int s = lane < nw ? wsum[lane] : 0;
      for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up(s, o, 32);
        if (lane >= o) s += n;
      }
      if (lane < nw) wsum[lane] = s;
    }
    __syncthreads();
    const int excl = carry + (wave ? wsum[wave - 1] : 0) + v - f;
    if (t < T) {
      tok2u[t] = f ? excl : -1;
      if (f) u2t[excl] = t;
      if (f && u2p)
        for (int j = 0; j < K; ++j) {
          const int p = pidx[size_t(t) * K + j];
          u2p[size_t(excl) * K + j] = p >= r0 ? p : -1;
        }
    }
    __syncthreads();
    if (tid == int(blockDim.x) - 1) carry = excl + f;
    __syncthreads();
  }
  if (tid == 0) plan->n_remote_tokens = carry;
  for (int p = r0 + tid; p < P; p += blockDim.x) p2u[p] = tok2u[eidx[p]];
  __threadfence_system();
}

// Block-stride over the compact tokens; each thread sums 8 columns (16-byte loads / stores).
__global__ __launch_bounds__(256) void remote_combine_kernel(const uint16_t* pair_out, const int32_t* u2p,
                                                             const MoePlan* plan, uint16_t* dst, int K, int dim) {
  const int U = plan->n_remote_tokens, cols = dim / 8;
  for (int u = blockIdx.x; u < U; u += gridDim.x)
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      float acc[8] = {};
      for (int j = 0; j < K; ++j) {
        const int p = u2p[size_t(u) * K + j];
        if (p < 0) continue;
        const uint4 r = reinterpret_cast<const uint4*>(pair_out + size_t(p) * dim)[c];
        const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          acc[2 * k] += bf16_to_f32(uint16_t(w[k]));
          acc[2 * k + 1] += bf16_to_f32(uint16_t(w[k] >> 16));
        }
      }
      uint32_t o[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) o[k] = uint32_t(f32_to_bf16(acc[2 * k])) | (uint32_t(f32_to_bf16(acc[2 * k + 1])) << 16);
      reinterpret_cast<uint4*>(dst + size_t(u) * dim)[c] = make_uint4(o[0], o[1], o[2], o[3]);
    }
  __threadfence_system();
}

__global__ void copy_remote_partial_kernel(const uint16_t* src, uint16_t* dst, const MoePlan* plan, int dim) {
  const size_t n = size_t(plan->n_remote_tokens) * dim / 8;
  const uint4* s = reinterpret_cast<const uint4*>(src);
  uint4* d = reinterpret_cast<uint4*>(dst + size_t(plan->remote0) * dim);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) d[i] = s[i];
}

// One block per (token, slice of dim): columns are independent.
__global__ void moe_combine_partial_kernel(const uint16_t* pair_out, const int32_t* pidx, const MoePlan* plan,
                                           const int32_t* tok2u, const uint16_t* shared, uint16_t* out, int K,
                                           int dim, int per) {
  const int t = blockIdx.x, r0 = plan->remote0, u = tok2u[t];
  const int d1 = min(dim, int(blockIdx.y + 1) * per);
  for (int d = blockIdx.y * per + threadIdx.x; d < d1; d += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
      const int p = pidx[size_t(t) * K + j];
      if (p < r0) acc += bf16_to_f32(pair_out[size_t(p) * dim + d]);
    }
    if (u >= 0) acc += bf16_to_f32(pair_out[size_t(r0 + u) * dim + d]);
    acc += bf16_to_f32(shared[size_t(t) * dim + d]);
    out[size_t(t) * dim + d] = f32_to_bf16(acc);
  }
}

// Wave = one 128-value segment of one compact token (4 values per lane): the scale as
// fake_quant_fp8_block computes it, the codes packed into one 4-byte store per lane (a 128-byte run per
// wave). Grid-stride over (token, segment) with one system-scope fence per block at the end: a fence per
// segment (131k blocks per layer) made the kernel wait on bus round trips.
__global__ __launch_bounds__(256) void quant_remote_tokens_kernel(const uint16_t* x, int dim, const int32_t* u2t,
                                                                  const MoePlan* plan, uint8_t* q, uint8_t* s) {
  const int lane = threadIdx.x & 31, nwave = blockDim.x >> 5, segs = dim / 128;
  const size_t n = size_t(plan->n_remote_tokens) * segs;
  for (size_t w = size_t(blockIdx.x) * nwave + (threadIdx.x >> 5); w < n; w += size_t(gridDim.x) * nwave) {
    const int u = int(w / segs), seg = int(w % segs);
    const uint2 raw = reinterpret_cast<const uint2*>(x + size_t(u2t[u]) * dim + seg * 128)[lane];
    const float v[4] = {bf16_to_f32(uint16_t(raw.x)), bf16_to_f32(uint16_t(raw.x >> 16)), bf16_to_f32(uint16_t(raw.y)),
                        bf16_to_f32(uint16_t(raw.y >> 16))};
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    a = fmaxf(warp_max(a), 1e-4f);
    const uint8_t e = pow2_ceil_e8m0(a * (1.0f / 448.0f));
    const float sc = e8m0_to_f32(e);
    uint32_t packed = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) packed |= uint32_t(f32_to_e4m3(fminf(fmaxf(v[k] / sc, -448.0f), 448.0f))) << (8 * k);
    reinterpret_cast<uint32_t*>(q + size_t(u) * dim + seg * 128)[lane] = packed;
    if (lane == 0) s[size_t(u) * segs + seg] = e;
  }
  __threadfence_system();
}

// moe_count + moe_plan + moe_scatter + remote_tokens + stage_remote for a step of <= 32 tokens, in one
// block of 256 threads: the counts, offsets and fill counters live in LDS (count / off / fill are left
// untouched), every output is what the five kernels write (the rows within an expert in another arrival
// order, as with their atomics).
__global__ __launch_bounds__(256) void moe_dispatch_small_kernel(MoeSmallArgs a) {
  __shared__ int scnt[256], soff[256], sfill[256];
  __shared__ int sc[4][256];
  __shared__ int stok[32];
  const int tid = threadIdx.x, lane = tid & 31;
  const int E = a.E, K = a.K, T = a.T;
  scnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < T * K; i += 256) {
    const int e = a.sel[i];
    atomicAdd(&scnt[e], 1);
    if (a.hist) atomicAdd(&a.hist[e], 1u);
  }
  __syncthreads();
  // moe_plan_kernel
  const int e = tid;
  const bool in = e < E, loc = in && a.resident[e] != 0;
  const int c = in ? scnt[e] : 0, nt = (c + a.tile_rows - 1) / a.tile_rows;
  const int v[4] = {loc ? c : 0, in && !loc ? c : 0, loc ? nt : 0, in && !loc ? nt : 0};
#pragma unroll
  for (int k = 0; k < 4; ++k) sc[k][e] = v[k];
  __syncthreads();
  for (int d = 1; d < 256; d <<= 1) {
    int add[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) add[k] = e >= d ? sc[k][e - d] : 0;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; ++k) sc[k][e] += add[k];
    __syncthreads();
  }
  const int remote0 = sc[0][255];
  if (in) {
    const int row = loc ? sc[0][e] - v[0] : remote0 + sc[1][e] - v[1];
    const int tile = loc ? sc[2][e] - v[2] : sc[3][e] - v[3];
    Exl3Tile* dst = loc ? a.lt : a.rtl;
    soff[e] = row;
    for (int i = 0; i < nt && tile + i < a.max_tiles; ++i)
      dst[tile + i] = Exl3Tile{e, row + i * a.tile_rows, min(a.tile_rows, c - i * a.tile_rows)};
  }
  sfill[tid] = 0;
  if (tid == 0) {
    a.plan->P = a.P;
    a.plan->remote0 = remote0;
    a.plan->n_local_tiles = min(sc[2][255], a.max_tiles);
    a.plan->n_remote_tiles = min(sc[3][255], a.max_tiles);
    a.plan->n_remote_tokens = 0;
  }
  __syncthreads();
  // moe_scatter_kernel
  if (tid < T) {
    const int t = tid;
    int es[8];
    float ws[8];
    for (int j = 0; j < K; ++j) {
      es[j] = a.sel[size_t(t) * K + j];
      ws[j] = a.wt[size_t(t) * K + j];
    }
    for (int x = 1; x < K; ++x)
      for (int y = x; y > 0 && es[y - 1] > es[y]; --y) {
        const int te = es[y];
        es[y] = es[y - 1];
        es[y - 1] = te;
        const float tw = ws[y];
        ws[y] = ws[y - 1];
        ws[y - 1] = tw;
      }
    for (int j = 0; j < K; ++j) {
      const int ej = es[j];
      const int p = soff[ej] + atomicAdd(&sfill[ej], 1);
      a.eidx[p] = t;
      a.ewt[p] = ws[j];
      a.row_expert[p] = ej;
      a.pidx[size_t(t) * K + j] = p;
    }
  }
  __syncthreads();
  if (!a.remote) return;
  // remote_tokens_kernel (one wave: T <= 32)
  if (a.qin && tid < 32) {
    const int t = lane;
    int f = 0;
    if (t < T)
      for (int j = 0; j < K; ++j) f |= a.pidx[size_t(t) * K + j] >= remote0;
    int incl = f;
    for (int o = 1; o < 32; o <<= 1) {
      const int n = __shfl_up(incl, o, 32);
      if (lane >= o) incl += n;
    }
    const int excl = incl - f;
    if (t < T) {
      a.tok2u[t] = f ? excl : -1;
      stok[t] = f ? excl : -1;
      if (f) a.u2t[excl] = t;
      if (f && a.u2p_dst)
        for (int j = 0; j < K; ++j) {
          const int p = a.pidx[size_t(t) * K + j];
          a.u2p_dst[size_t(excl) * K + j] = p >= remote0 ? p : -1;
        }
    }
    if (lane == 31) a.plan->n_remote_tokens = incl;
  }
  __syncthreads();
  if (a.qin)
    for (int p = remote0 + tid; p < a.P; p += 256) a.p2u_dst[p] = stok[a.eidx[p]];
  // stage_remote_kernel
  const MoePlan pl = *a.plan;
  if (tid == 0) *a.plan_dst = pl;
  for (int i = tid; i < pl.n_remote_tiles; i += 256) a.tiles_dst[i] = a.rtl[i];
  for (int p = pl.remote0 + tid; p < pl.P; p += 256) {
    a.w_dst[p] = a.ewt[p];
    a.e_dst[p] = a.row_expert[p];
  }
  __threadfence_system();
}

}  // namespace

void remote_tokens(const int32_t* pidx, const int32_t* eidx, int T, int K, MoePlan* plan, int32_t* tok2u,
                   int32_t* u2t, int32_t* p2u_dst, int32_t* u2p_dst, hipStream_t st) {
  remote_tokens_kernel<<<1, 1024, 0, st>>>(pidx, eidx, T, K, plan, tok2u, u2t, p2u_dst, u2p_dst);
  UH_CHECK_LAUNCH();
}

void remote_combine(const uint16_t* pair_out, const int32_t* u2p, const MoePlan* plan, uint16_t* dst, int K, int dim,
                    int max_tokens, hipStream_t st) {
  if (max_tokens <= 0) return;
  if (dim % 8) throw std::runtime_error("remote_combine: dim % 8");
  remote_combine_kernel<<<std::min(max_tokens, 512), 256, 0, st>>>(pair_out, u2p, plan, dst, K, dim);
  UH_CHECK_LAUNCH();
}

void copy_remote_partial(const uint16_t* src, uint16_t* dst, const MoePlan* plan, int dim, int max_tokens,
                         hipStream_t st) {
  const size_t units = size_t(std::max(max_tokens, 1)) * dim / 8;
  copy_remote_partial_kernel<<<int(std::min<size_t>((units + 255) / 256, 1024)), 256, 0, st>>>(src, dst, plan, dim);
  UH_CHECK_LAUNCH();
}

void moe_combine_partial(const uint16_t* pair_out, const int32_t* pidx, const MoePlan* plan, const int32_t* tok2u,
                         const uint16_t* shared, uint16_t* out, int T, int K, int dim, hipStream_t st) {
  if (T <= 0) return;
  const int sl = row_slices(T, dim);
  moe_combine_partial_kernel<<<dim3(T, sl), 256, 0, st>>>(pair_out, pidx, plan, tok2u, shared, out, K, dim,
                                                          (dim + sl - 1) / sl);
  UH_CHECK_LAUNCH();
}

void quant_remote_tokens(const uint16_t* x, int dim, const int32_t* u2t, const MoePlan* plan, uint8_t* q_dst,
                         uint8_t* s_dst, int max_tokens, hipStream_t st) {
  if (max_tokens <= 0) return;
  if (dim % 128) throw std::runtime_error("quant_remote_tokens: dim % 128");
  const size_t waves = size_t(max_tokens) * (dim / 128);
  const int blocks = int(std::min<size_t>((waves + 7) / 8, 1024));
  quant_remote_tokens_kernel<<<blocks, 256, 0, st>>>(x, dim, u2t, plan, q_dst, s_dst);
  UH_CHECK_LAUNCH();
}

void stage_remote(const MoePlan* plan, const Exl3Tile* rtiles, const float* ewt, const int32_t* row_expert,
                  MoePlan* plan_dst, Exl3Tile* tiles_dst, float* w_dst, int32_t* e_dst, int max_rows, hipStream_t st) {
  stage_remote_kernel<<<std::max(1, std::min((max_rows + 255) / 256, 64)), 256, 0, st>>>(plan, rtiles, ewt, row_expert,
                                                                                       plan_dst, tiles_dst, w_dst, e_dst);
  UH_CHECK_LAUNCH();
}

void moe_dispatch_small(const MoeSmallArgs& a, hipStream_t st) {
  if (a.T > kMoeSmallMaxTokens || a.K > 8 || a.E > 256) throw std::runtime_error("moe_dispatch_small: shape");
  moe_dispatch_small_kernel<<<1, 256, 0, st>>>(a);
  UH_CHECK_LAUNCH();
}

void moe_count(const int32_t* sel, int T, int K, int32_t* count, int E, hipStream_t st, uint32_t* hist) {
  const int n = T * K;
  moe_count_kernel<<<(n + 255) / 256, 256, 0, st>>>(sel, n, count, hist);
  UH_CHECK_LAUNCH();
}

void moe_plan(int32_t* count, const uint8_t* resident, int E, int P, MoePlan* plan, Exl3Tile* local_tiles,
              Exl3Tile* remote_tiles, int32_t* off, int32_t* fill, int max_tiles, hipStream_t st, int tile_rows) {
  if (E > 256) throw std::runtime_error("moe_plan: more than 256 experts");
  if (tile_rows < 1 || tile_rows > kExl3TileRows) throw std::runtime_error("moe_plan: tile rows");
  moe_plan_kernel<<<1, 256, 0, st>>>(count, resident, E, P, plan, local_tiles, remote_tiles, off, fill, max_tiles,
                                     tile_rows);
  UH_CHECK_LAUNCH();
}

void moe_scatter(const int32_t* sel, const float* wt, int T, int K, const int32_t* off, int32_t* fill, int32_t* eidx,
                 float* ewt, int32_t* row_expert, int32_t* pidx, hipStream_t st) {
  moe_scatter_kernel<<<(T + 63) / 64, 64, 0, st>>>(sel, wt, T, K, off, fill, eidx, ewt, row_expert, pidx);
  UH_CHECK_LAUNCH();
}

void copy_remote_rows(const uint16_t* src, uint16_t* dst, const MoePlan* plan, int dim, int max_rows, hipStream_t st) {
  const size_t units = size_t(max_rows) * dim / 8;
  const int blocks = int(std::min<size_t>((units + 255) / 256, 1024));
  copy_remote_rows_kernel<<<std::max(blocks, 1), 256, 0, st>>>(src, dst, plan, dim);
  UH_CHECK_LAUNCH();
}

void copy_remote_meta(const float* ewt, const int32_t* row_expert, float* w_dst, int32_t* e_dst, const MoePlan* plan,
                      int max_rows, hipStream_t st) {
  copy_remote_meta_kernel<<<std::max(1, std::min((max_rows + 255) / 256, 256)), 256, 0, st>>>(ewt, row_expert, w_dst,
                                                                                               e_dst, plan);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

// Reference attention-side kernels: compressor rows, paged stores, Lightning Indexer scores and top-k,
// sparse attention with sink, window ring.
#include "kernels/dev.hip.hpp"
#include "kernels/ref.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

inline int cdiv(size_t a, int b) { return int((a + b - 1) / b); }

__device__ __forceinline__ const bf16* paged_row(const PagedRows& p, int r) {
  const int page = r / p.rows_per_page, slot = r % p.rows_per_page;
  return p.pool + (size_t(p.table[page]) * p.rows_per_page + slot) * p.row_stride;
}

// ---- compressor ----
// One thread per (row, channel c in [0, d)).
__global__ void compress_rows_kernel(const float* pkv, const float* psc, const float* rkv, const float* rsc, float* out,
                                     int s0, int n, int row0, int nrows, int ratio, int d, bool overlap, int ring) {
  const size_t gid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (gid >= size_t(nrows) * d) return;
  const int i = row0 + int(gid / d), c = int(gid % d);
  const int width = overlap ? 2 * d : d;
  auto fetch = [&](int p, int ch, float& kv, float& sc) {
    if (p < 0) {
      kv = 0.0f;
      sc = -INFINITY;
    } else if (p >= s0) {
      kv = pkv[size_t(p - s0) * width + ch];
      sc = psc[size_t(p - s0) * width + ch];
    } else {
      kv = rkv[size_t(p % ring) * width + ch];
      sc = rsc[size_t(p % ring) * width + ch];
    }
  };
  constexpr int kMax = 128;
  float kv[kMax], sc[kMax];
  int m = 0;
  if (overlap) {
    for (int r = 0; r < ratio; ++r, ++m) fetch((i - 1) * ratio + r, c, kv[m], sc[m]);
    for (int r = 0; r < ratio; ++r, ++m) fetch(i * ratio + r, d + c, kv[m], sc[m]);
  } else {
    for (int r = 0; r < ratio; ++r, ++m) fetch(i * ratio + r, c, kv[m], sc[m]);
  }
  float mx = -INFINITY;
  for (int e = 0; e < m; ++e) mx = fmaxf(mx, sc[e]);
  float sum = 0.0f;
  for (int e = 0; e < m; ++e) {
    sc[e] = expf(sc[e] - mx);
    sum += sc[e];
  }
  float acc = 0.0f;
  for (int e = 0; e < m; ++e) acc += kv[e] * (sc[e] / sum);
  out[size_t(i - row0) * d + c] = acc;
}

__global__ void add_ape_kernel(float* score, const float* ape, int s0, int n, int ratio, int width) {
  const size_t gid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (gid >= size_t(n) * width) return;
  const int t = int(gid / width), c = int(gid % width);
  score[gid] += ape[size_t((s0 + t) % ratio) * width + c];
}

__global__ void ring_update_kernel(const float* pkv, const float* psc, float* rkv, float* rsc, int s0, int n, int ring,
                                   int width) {
  const int first = max(0, n - ring);
  const size_t gid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (gid >= size_t(n - first) * width) return;
  const int t = first + int(gid / width), c = int(gid % width);
  const int slot = (s0 + t) % ring;
  rkv[size_t(slot) * width + c] = pkv[size_t(t) * width + c];
  rsc[size_t(slot) * width + c] = psc[size_t(t) * width + c];
}

__global__ void store_rows_kernel(const bf16* rows, int n, int width, PagedRows dst, int row0) {
  const size_t gid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (gid >= size_t(n) * width) return;
  const int r = int(gid / width), c = int(gid % width);
  const_cast<bf16*>(paged_row(dst, row0 + r))[c] = rows[gid];
}

// ---- indexer ----
// Block: one query, 32 keys; 256 threads = 8 waves; each wave scores 4 keys (one head per lane pass).
__global__ void indexer_scores_kernel(const bf16* q, const bf16* w, PagedFp4 keys, const int32_t* pos, float* out,
                                      int H, int D, int nkeys, int ratio) {
  extern __shared__ float qs[];  // [H * D] + [H]
  float* ws = qs + H * D;
  const int t = blockIdx.y;
  for (int i = threadIdx.x; i < H * D; i += blockDim.x) qs[i] = bf16_to_f32(q[size_t(t) * H * D + i]);
  for (int i = threadIdx.x; i < H; i += blockDim.x) ws[i] = bf16_to_f32(w[size_t(t) * H + i]);
  __syncthreads();
  const int visible = (pos[t] + 1) / ratio;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  for (int kk = 0; kk < 4; ++kk) {
    const int j = blockIdx.x * 32 + warp * 4 + kk;
    if (j >= nkeys) continue;
    if (j >= visible) {
      if (lane == 0) out[size_t(t) * nkeys + j] = -INFINITY;
      continue;
    }
    // Key row j from the fp4 pool: the values the reference's fake_quant_fp4 produced (exact in bf16).
    const int page = keys.table[j / kFp4RowsPerPage], slot = j % kFp4RowsPerPage;
    const uint8_t* kc = keys.pool + size_t(page) * kFp4PageBytes + size_t(slot) * kFp4RowBytes;
    const uint8_t* ks = keys.pool + size_t(page) * kFp4PageBytes + kFp4RowsPerPage * kFp4RowBytes + size_t(slot) * 4;
    float kf[4];
    const int per = D / 32;  // D = 128 -> 4 dims per lane
    for (int e = 0; e < per; ++e) {
      const int c = lane * per + e;
      const uint32_t nib = (c & 1) ? kc[c / 2] >> 4 : kc[c / 2] & 15;
      kf[e] = round_bf16(e2m1_to_f32(nib) * e8m0_to_f32(ks[c / 32]));
    }
    float total = 0.0f;
    for (int h = 0; h < H; ++h) {
      float dot = 0.0f;
      for (int e = 0; e < per; ++e) dot += qs[h * D + lane * per + e] * kf[e];
      dot = warp_sum(dot);
      const float relu = fmaxf(round_bf16(dot), 0.0f);
      total += round_bf16(relu * ws[h]);
    }
    if (lane == 0) out[size_t(t) * nkeys + j] = round_bf16(total);
  }
}

// Order-preserving map of fp32 -> uint32 (larger float -> larger key).
__device__ __forceinline__ uint32_t sort_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One block (256 threads) per query row. Radix select on the 32-bit key (4 passes of 8 bits), then
// collect: keys above the threshold, plus threshold ties by lowest index; -inf never selected.
// Output sorted by (score desc, index asc) via an LDS bitonic sort of up to 1024 entries.
__global__ void topk_rows_kernel(const float* scores, int n, int k, int32_t* idx) {
  __shared__ uint32_t hist[256];
  __shared__ uint32_t prefix_sel, need_sel;
  __shared__ uint32_t sel_keys[1024];
  __shared__ int32_t sel_idx[1024];
  __shared__ uint32_t count_above, tie_taken;
  const float* row = scores + size_t(blockIdx.x) * n;
  const uint32_t neg_inf_key = sort_key(-INFINITY);
  // Count valid entries.
  __shared__ uint32_t nvalid;
  if (threadIdx.x == 0) nvalid = 0;
  __syncthreads();
  uint32_t local = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) local += sort_key(row[i]) > neg_inf_key;
  atomicAdd(&nvalid, local);
  __syncthreads();
  const uint32_t want = min(uint32_t(k), nvalid);
  uint32_t prefix = 0, need = want;  // find the want-th largest key
  if (want > 0) {
    for (int pass = 3; pass >= 0; --pass) {
      for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
      __syncthreads();
      const uint32_t mask_hi = pass == 3 ? 0u : (0xffffffffu << ((pass + 1) * 8));
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t key = sort_key(row[i]);
        if (key <= neg_inf_key) continue;
        if ((key & mask_hi) != (prefix & mask_hi)) continue;
        atomicAdd(&hist[(key >> (pass * 8)) & 0xff], 1u);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        uint32_t cum = 0;
        int b = 255;
        for (; b > 0; --b) {
          if (cum + hist[b] >= need) break;
          cum += hist[b];
        }
        prefix_sel = prefix | (uint32_t(b) << (pass * 8));
        need_sel = need - cum;
      }
      __syncthreads();
      prefix = prefix_sel;
      need = need_sel;
    }
  }
  // prefix = threshold key T; need = number of ties at T to take.
  if (threadIdx.x == 0) {
    count_above = 0;
    tie_taken = 0;
  }
  __syncthreads();
  // Keys strictly above the threshold (order irrelevant, sorted below).
  for (int i = threadIdx.x; i < n && want > 0; i += blockDim.x) {
    const uint32_t key = sort_key(row[i]);
    if (key > prefix && key > neg_inf_key) {
      const uint32_t slot = atomicAdd(&count_above, 1u);
      sel_keys[slot] = key;
      sel_idx[slot] = i;
    }
  }
  __syncthreads();
  // Ties at the threshold, lowest index first: serial scan by one wave in index order.
  if (threadIdx.x < 32 && want > 0) {
    const uint32_t base = count_above;
    uint32_t taken = 0;
    for (int i0 = 0; i0 < n && taken < need; i0 += 32) {
      const int i = i0 + threadIdx.x;
      const bool tie = i < n && sort_key(row[i]) == prefix;
      const uint64_t ballot = __ballot(tie);
      const uint32_t before = __popcll(ballot & ((1ull << threadIdx.x) - 1));
      if (tie && taken + before < need) {
        sel_keys[base + taken + before] = prefix;
        sel_idx[base + taken + before] = i;
      }
      taken += __popcll(ballot);
    }
  }
  __syncthreads();
  // Pad and sort: key desc, index asc. Size: next power of two >= k (<= 1024).
  int size = 1;
  while (size < k) size <<= 1;
  for (int i = threadIdx.x; i < size; i += blockDim.x)
    if (uint32_t(i) >= want) {
      sel_keys[i] = 0;
      sel_idx[i] = 0x7fffffff;
    }
  __syncthreads();
  for (int len = 2; len <= size; len <<= 1)
    for (int stride = len >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const bool desc = (i & len) == 0;
          // "a before b" if key larger, or equal key and smaller index.
          const bool i_first = sel_keys[i] > sel_keys[j] || (sel_keys[i] == sel_keys[j] && sel_idx[i] < sel_idx[j]);
          if (desc ? !i_first : i_first) {
            const uint32_t tk = sel_keys[i];
            sel_keys[i] = sel_keys[j];
            sel_keys[j] = tk;
            const int32_t ti = sel_idx[i];
            sel_idx[i] = sel_idx[j];
            sel_idx[j] = ti;
          }
        }
      }
      __syncthreads();
    }
  for (int i = threadIdx.x; i < k; i += blockDim.x)
    idx[size_t(blockIdx.x) * k + i] = uint32_t(i) < want ? sel_idx[i] : -1;
}

// ---- sparse attention ----
// One block (256 threads) per (query, head). Entries: 128 window slots, then compressed rows.
__global__ void sparse_attn_kernel(const bf16* q, const int32_t* pos, AttnKv kv, const float* sink, bf16* o, int H,
                                   int D, float scale) {
  constexpr int kBlock = 64;
  __shared__ float qs[512];
  __shared__ float s[kBlock];
  __shared__ float p[kBlock];
  __shared__ const bf16* rows[kBlock];
  __shared__ float m_run, l_run, f_scale;
  const int t = blockIdx.x / H, h = blockIdx.x % H;
  const int p_t = pos[t];
  for (int i = threadIdx.x; i < D; i += blockDim.x) qs[i] = bf16_to_f32(q[(size_t(t) * H + h) * D + i]);
  const int visible = kv.ratio ? (p_t + 1) / kv.ratio : 0;
  const int ncomp = kv.ratio == 0 ? 0 : (kv.ncomp > 0 ? kv.ncomp : visible);
  const int total = 128 + ncomp;
  if (threadIdx.x == 0) {
    m_run = -INFINITY;
    l_run = 0.0f;
  }
  float acc[2] = {0.0f, 0.0f};  // dims threadIdx.x and threadIdx.x + 256 (D = 512)
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  for (int e0 = 0; e0 < total; e0 += kBlock) {
    __syncthreads();
    if (threadIdx.x < kBlock) {
      const int e = e0 + threadIdx.x;
      const bf16* r = nullptr;
      if (e < 128) {
        // Ascending positions from max(0, p - 127), unused slots at the end (reference order).
        const int wp = max(0, p_t - 127) + e;
        if (wp <= p_t) r = wp >= kv.s0 ? kv.chunk_kv + size_t(wp - kv.s0) * D : kv.ring + size_t(wp % 128) * D;
      } else if (e < total) {
        const int ci = e - 128;
        const int row = kv.ncomp > 0 ? kv.comp_idx[size_t(t) * kv.ncomp + ci] : ci;
        if (row >= 0 && row < visible) r = paged_row(kv.comp, row);
      }
      rows[threadIdx.x] = r;
    }
    __syncthreads();
    // Scores: wave w handles entries w*8 .. w*8+7; each lane 16 dims.
    for (int kk = 0; kk < 8; ++kk) {
      const int e = warp * 8 + kk;
      const bf16* r = rows[e];
      float dot = 0.0f;
      if (r)
        for (int i = 0; i < D / 32; ++i) dot += qs[lane * (D / 32) + i] * bf16_to_f32(r[lane * (D / 32) + i]);
      dot = warp_sum(dot);
      if (lane == 0) s[e] = r ? dot * scale : -INFINITY;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bm = m_run;
      for (int e = 0; e < kBlock; ++e) bm = fmaxf(bm, s[e]);
      const float fs = expf(m_run - bm);
      float sum = 0.0f;
      for (int e = 0; e < kBlock; ++e) {
        const float pe = expf(s[e] - bm);
        sum += pe;
        p[e] = round_bf16(pe);
      }
      l_run = l_run * fs + sum;
      m_run = bm;
      f_scale = fs;
    }
    __syncthreads();
    const float fs = f_scale;
    for (int j = 0; j < 2; ++j) {
      const int dim = threadIdx.x + j * 256;
      float a = acc[j] * fs;
      for (int e = 0; e < kBlock; ++e)
        if (rows[e]) a += p[e] * bf16_to_f32(rows[e][dim]);
      acc[j] = a;
    }
  }
  __syncthreads();
  const float denom = l_run + expf(sink[h] - m_run);
  for (int j = 0; j < 2; ++j) {
    const int dim = threadIdx.x + j * 256;
    o[(size_t(t) * H + h) * D + dim] = f32_to_bf16(acc[j] / denom);
  }
}

__global__ void window_update_kernel(const bf16* chunk_kv, bf16* ring, int s0, int n, int D) {
  const int first = max(0, n - 128);
  const size_t gid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (gid >= size_t(n - first) * D) return;
  const int t = first + int(gid / D), c = int(gid % D);
  ring[size_t((s0 + t) % 128) * D + c] = chunk_kv[size_t(t) * D + c];
}

}  // namespace

void compress_rows(const float* proj_kv, const float* proj_score, const float* ring_kv, const float* ring_score,
                   float* out, int s0, int n, int row0, int nrows, int ratio, int d, bool overlap, int ring,
                   hipStream_t st) {
  if (nrows <= 0) return;
  if ((overlap ? 2 : 1) * ratio > 128) throw std::runtime_error("compress_rows: window too large");
  compress_rows_kernel<<<cdiv(size_t(nrows) * d, 256), 256, 0, st>>>(proj_kv, proj_score, ring_kv, ring_score, out, s0,
                                                                      n, row0, nrows, ratio, d, overlap, ring);
  UH_CHECK_LAUNCH();
}
void add_ape(float* score, const float* ape, int s0, int n, int ratio, int width, hipStream_t st) {
  add_ape_kernel<<<cdiv(size_t(n) * width, 256), 256, 0, st>>>(score, ape, s0, n, ratio, width);
  UH_CHECK_LAUNCH();
}
void ring_update(const float* proj_kv, const float* proj_score, float* ring_kv, float* ring_score, int s0, int n,
                 int ring, int width, hipStream_t st) {
  const int cnt = n - std::max(0, n - ring);
  if (cnt <= 0) return;
  ring_update_kernel<<<cdiv(size_t(cnt) * width, 256), 256, 0, st>>>(proj_kv, proj_score, ring_kv, ring_score, s0, n,
                                                                      ring, width);
  UH_CHECK_LAUNCH();
}
void store_rows(const bf16* rows, int n, int width, PagedRows dst, int row0, hipStream_t st) {
  if (n <= 0) return;
  store_rows_kernel<<<cdiv(size_t(n) * width, 256), 256, 0, st>>>(rows, n, width, dst, row0);
  UH_CHECK_LAUNCH();
}
void indexer_scores(const bf16* q, const bf16* w, PagedFp4 keys, const int32_t* pos, float* out, int T, int H, int D,
                    int nkeys, int ratio, hipStream_t st) {
  if (D != 128) throw std::runtime_error("indexer_scores: D must be 128");
  dim3 grid(cdiv(nkeys, 32), T);
  const size_t lds = (size_t(H) * D + H) * sizeof(float);
  indexer_scores_kernel<<<grid, 256, lds, st>>>(q, w, keys, pos, out, H, D, nkeys, ratio);
  UH_CHECK_LAUNCH();
}
void topk_rows(const float* scores, int T, int n, int k, int32_t* idx, hipStream_t st) {
  if (k > 1024) throw std::runtime_error("topk_rows: k > 1024");
  topk_rows_kernel<<<T, 256, 0, st>>>(scores, n, k, idx);
  UH_CHECK_LAUNCH();
}
void sparse_attn(const bf16* q, const int32_t* pos, AttnKv kv, const float* sink, bf16* o, int T, int H, int D,
                 float scale, hipStream_t st) {
  if (D != 512) throw std::runtime_error("sparse_attn: D must be 512");
  sparse_attn_kernel<<<T * H, 256, 0, st>>>(q, pos, kv, sink, o, H, D, scale);
  UH_CHECK_LAUNCH();
}
void window_update(const bf16* chunk_kv, bf16* ring, int s0, int n, int D, hipStream_t st) {
  const int cnt = n - std::max(0, n - 128);
  if (cnt <= 0) return;
  window_update_kernel<<<cdiv(size_t(cnt) * D, 256), 256, 0, st>>>(chunk_kv, ring, s0, n, D);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

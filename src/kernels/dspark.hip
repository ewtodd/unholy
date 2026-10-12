// DSpark drafter kernels (kernels/dspark.hpp).
#include "kernels/dev.hip.hpp"
#include "kernels/dspark.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

constexpr int kMaxRequests = 16;  // markov_step: requests per launch
constexpr int kRowsPerBlock = 128;  // markov_step: vocab rows per block (8 waves x 16)

__global__ void hc_mean_rows_kernel(const bf16* x, int hc, int D, const int32_t* rows, bf16* out, int ldo) {
  const int r = blockIdx.x;
  const bf16* xr = x + size_t(rows[r]) * hc * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float s = 0.0f;
    for (int j = 0; j < hc; ++j) s += bf16_to_f32(xr[size_t(j) * D + i]);
    out[size_t(r) * ldo + i] = f32_to_bf16(s / float(hc));
  }
}

__global__ void scatter_window_kernel(const bf16* kv, int D, const int32_t* slots, const int32_t* pos, bf16* ring) {
  const int r = blockIdx.x;
  bf16* dst = ring + (size_t(slots[r]) * 128 + size_t(pos[r] % 128)) * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) dst[i] = kv[size_t(r) * D + i];
}

// Bias and per-block argmax: wave w of block b takes vocab rows b * 128 + w * 16 .. +15, one row at a
// time with the K = 256 dims across the lanes (8 each, one 16-byte load).
__global__ __launch_bounds__(256) void markov_bias_kernel(const bf16* w1, const bf16* w2, int V, float* logits, int B,
                                                          int32_t* tok, bf16* emb, int R, int i, float* part_v,
                                                          int32_t* part_i) {
  constexpr int K = 256;
  __shared__ float e[kMaxRequests][K];
  __shared__ float bv[kMaxRequests][8];
  __shared__ int32_t bi[kMaxRequests][8];
  for (int idx = threadIdx.x; idx < R * K; idx += blockDim.x) {
    const int r = idx / K, k = idx % K;
    const bf16 v = w1[size_t(tok[r * (B + 1) + i]) * K + k];
    e[r][k] = bf16_to_f32(v);
    if (blockIdx.x == 0) emb[size_t(r * B + i) * K + k] = v;
  }
  __syncthreads();
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float best[kMaxRequests];
  int32_t besti[kMaxRequests];
  for (int r = 0; r < R; ++r) {
    best[r] = -INFINITY;
    besti[r] = 0x7fffffff;
  }
  for (int j = 0; j < 16; ++j) {
    const int v = blockIdx.x * kRowsPerBlock + wave * 16 + j;
    if (v >= V) break;
    const uint4 raw = *reinterpret_cast<const uint4*>(w2 + size_t(v) * K + lane * 8);
    const bf16* wv = reinterpret_cast<const bf16*>(&raw);
    for (int r = 0; r < R; ++r) {
      float d = 0.0f;
#pragma unroll
      for (int k = 0; k < 8; ++k) d = fmaf(bf16_to_f32(wv[k]), e[r][lane * 8 + k], d);
      d = warp_sum(d);
      if (lane == 0) {
        float& l = logits[size_t(r * B + i) * V + v];
        l += d;
        if (l > best[r]) {  // rows ascend, so the first maximum keeps the lowest index
          best[r] = l;
          besti[r] = v;
        }
      }
    }
  }
  if (lane == 0)
    for (int r = 0; r < R; ++r) {
      bv[r][wave] = best[r];
      bi[r][wave] = besti[r];
    }
  __syncthreads();
  if (int(threadIdx.x) < R) {
    const int r = threadIdx.x;
    float m = bv[r][0];
    int32_t mi = bi[r][0];
    for (int w = 1; w < 8; ++w)
      if (bv[r][w] > m || (bv[r][w] == m && bi[r][w] < mi)) {
        m = bv[r][w];
        mi = bi[r][w];
      }
    part_v[size_t(r) * gridDim.x + blockIdx.x] = m;
    part_i[size_t(r) * gridDim.x + blockIdx.x] = mi;
  }
}

__global__ void markov_argmax_kernel(const float* part_v, const int32_t* part_i, int nblocks, int32_t* tok, int B,
                                     int i) {
  __shared__ float sv[256];
  __shared__ int32_t si[256];
  const int r = blockIdx.x;
  float m = -INFINITY;
  int32_t mi = 0x7fffffff;
  for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
    const float v = part_v[size_t(r) * nblocks + b];
    const int32_t x = part_i[size_t(r) * nblocks + b];
    if (v > m || (v == m && x < mi)) {
      m = v;
      mi = x;
    }
  }
  sv[threadIdx.x] = m;
  si[threadIdx.x] = mi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (int(threadIdx.x) < s) {
      const float v = sv[threadIdx.x + s];
      const int32_t x = si[threadIdx.x + s];
      if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && x < si[threadIdx.x])) {
        sv[threadIdx.x] = v;
        si[threadIdx.x] = x;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) tok[r * (B + 1) + i + 1] = si[0];
}

__global__ void confidence_kernel(const bf16* x, int D, const bf16* emb, int K, const bf16* w, float* conf) {
  __shared__ float red[8];
  const int n = blockIdx.x;
  float s = 0.0f;
  for (int j = threadIdx.x; j < D + K; j += blockDim.x) {
    const float v = j < D ? bf16_to_f32(x[size_t(n) * D + j]) : bf16_to_f32(emb[size_t(n) * K + (j - D)]);
    s = fmaf(v, bf16_to_f32(w[j]), s);
  }
  s = warp_sum(s);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < int(blockDim.x / 32); ++i) t += red[i];
    conf[n] = t;
  }
}

__global__ void ring_copy_kernel(const RingDesc* descs, int slot, int s0, int p0, uint8_t* backup, size_t pos_bytes,
                                 bool restore) {
  const RingDesc d = descs[blockIdx.x];
  const int p = p0 + int(blockIdx.y);
  uint8_t* ring = d.base + size_t(slot) * d.slot_bytes + size_t(p % d.ring) * d.row_bytes;
  uint8_t* saved = backup + size_t(p - s0) * pos_bytes + d.off;
  const uint4* src = reinterpret_cast<const uint4*>(restore ? saved : ring);
  uint4* dst = reinterpret_cast<uint4*>(restore ? ring : saved);
  for (int i = int(threadIdx.x); i < d.row_bytes / 16; i += int(blockDim.x)) dst[i] = src[i];
}

}  // namespace

void hc_mean_rows(const bf16* x, int hc, int D, const int32_t* rows, int R, bf16* out, int ldo, hipStream_t st) {
  if (R <= 0) return;
  hc_mean_rows_kernel<<<R, 256, 0, st>>>(x, hc, D, rows, out, ldo);
  UH_CHECK_LAUNCH();
}

void scatter_window(const bf16* kv, int R, int D, const int32_t* slots, const int32_t* pos, bf16* ring,
                    hipStream_t st) {
  if (R <= 0) return;
  scatter_window_kernel<<<R, 256, 0, st>>>(kv, D, slots, pos, ring);
  UH_CHECK_LAUNCH();
}

void ring_backup(const RingDesc* descs, int nd, int slot, int s0, int n, uint8_t* backup, size_t pos_bytes,
                 hipStream_t st) {
  if (nd <= 0 || n <= 0) return;
  ring_copy_kernel<<<dim3(unsigned(nd), unsigned(n)), 256, 0, st>>>(descs, slot, s0, s0, backup, pos_bytes, false);
  UH_CHECK_LAUNCH();
}

void ring_restore(const RingDesc* descs, int nd, int slot, int s0, int from, int to, const uint8_t* backup,
                  size_t pos_bytes, hipStream_t st) {
  if (nd <= 0 || to <= from) return;
  ring_copy_kernel<<<dim3(unsigned(nd), unsigned(to - from)), 256, 0, st>>>(descs, slot, s0, from,
                                                                          const_cast<uint8_t*>(backup), pos_bytes, true);
  UH_CHECK_LAUNCH();
}

size_t markov_scratch_bytes(int V, int R) {
  return size_t(R) * size_t((V + kRowsPerBlock - 1) / kRowsPerBlock) * 8;
}

void markov_step(const bf16* w1, const bf16* w2, int V, int K, float* logits, int B, int32_t* tok, bf16* emb, int R,
                 int i, void* part, hipStream_t st) {
  if (K != 256) throw std::runtime_error("markov_step: rank 256 only");
  if (R <= 0) return;
  if (R > kMaxRequests) throw std::runtime_error("markov_step: too many requests");
  const int nb = (V + kRowsPerBlock - 1) / kRowsPerBlock;
  auto* pv = static_cast<float*>(part);
  auto* pi = reinterpret_cast<int32_t*>(pv + size_t(R) * nb);
  markov_bias_kernel<<<nb, 256, 0, st>>>(w1, w2, V, logits, B, tok, emb, R, i, pv, pi);
  UH_CHECK_LAUNCH();
  markov_argmax_kernel<<<R, 256, 0, st>>>(pv, pi, nb, tok, B, i);
  UH_CHECK_LAUNCH();
}

void dspark_confidence(const bf16* x, int D, const bf16* emb, int K, const bf16* w, float* conf, int N,
                       hipStream_t st) {
  if (N <= 0) return;
  confidence_kernel<<<N, 256, 0, st>>>(x, D, emb, K, w, conf);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

#include "kernels/dev.hip.hpp"
#include "kernels/indexer.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

inline int cdiv(size_t a, int b) { return int((a + b - 1) / b); }

// e2m1 nibble of a value already on the e2m1 grid (round_e2m1 output): sign bit 8, magnitude index
// 0..7 for 0, 0.5, 1, 1.5, 2, 3, 4, 6.
__device__ __forceinline__ uint32_t e2m1_code(float r) {
  const float a = fabsf(r);
  const uint32_t m = a < 2.0f ? uint32_t(a * 2.0f) : (a == 6.0f ? 7u : uint32_t(a) + 2u);
  return m | (signbit(r) ? 8u : 0u);
}

// 2 * e2m1 value of a nibble as a signed integer: 0, 1, 2, 3, 4, 6, 8, 12.
__device__ __forceinline__ int8_t e2m1_int2(uint32_t nib) {
  const uint32_t m = nib & 7;
  const int v = m < 5 ? int(m) : (m == 5 ? 6 : (m == 6 ? 8 : 12));
  return int8_t((nib & 8) ? -v : v);
}

__device__ __forceinline__ uint8_t* fp4_codes(const PagedFp4& p, int r) {
  const int page = p.table[r / kFp4RowsPerPage], slot = r % kFp4RowsPerPage;
  return p.pool + size_t(page) * kFp4PageBytes + size_t(slot) * kFp4RowBytes;
}
__device__ __forceinline__ uint8_t* fp4_scales(const PagedFp4& p, int r) {
  const int page = p.table[r / kFp4RowsPerPage], slot = r % kFp4RowsPerPage;
  return p.pool + size_t(page) * kFp4PageBytes + kFp4RowsPerPage * kFp4RowBytes + size_t(slot) * 4;
}

// One wave per (row, 32-block), a value per lane: fake_quant_fp4's amax / scale / rounding.
__device__ __forceinline__ void quant32(float v, uint32_t& code, uint8_t& se) {
  const float amax = fmaxf(warp_max(fabsf(v)), 6.0f * 1.1754943508222875e-38f);
  se = pow2_ceil_e8m0(amax * (1.0f / 6.0f));
  const float sc = e8m0_to_f32(se);
  code = e2m1_code(round_e2m1(fminf(fmaxf(v / sc, -6.0f), 6.0f)));
}

__global__ void store_rows_fp4_kernel(const bf16* rows, int n, PagedFp4 dst, int row0) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (wid >= n * 4) return;
  const int r = wid / 4, b = wid % 4;
  uint32_t code;
  uint8_t se;
  quant32(bf16_to_f32(rows[size_t(r) * 128 + b * 32 + lane]), code, se);
  const uint32_t hi = __shfl_down(code, 1, 32);
  if ((lane & 1) == 0) fp4_codes(dst, row0 + r)[b * 16 + lane / 2] = uint8_t(code | (hi << 4));
  if (lane == 0) fp4_scales(dst, row0 + r)[b] = se;
}

__global__ void load_rows_fp4_kernel(PagedFp4 src, int row0, int n, bf16* out) {
  const size_t gid = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (gid >= size_t(n) * 128) return;
  const int r = int(gid / 128), c = int(gid % 128);
  const uint8_t byte = fp4_codes(src, row0 + r)[c / 2];
  const float v = e2m1_to_f32((c & 1) ? byte >> 4 : byte & 15) * e8m0_to_f32(fp4_scales(src, row0 + r)[c / 32]);
  out[gid] = f32_to_bf16(v);
}

__global__ void quant_q_fp4_kernel(const bf16* q, int blocks, int8_t* qc, uint8_t* qs) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (wid >= blocks) return;  // wid = (t * H + h) * 4 + b
  uint32_t code;
  uint8_t se;
  quant32(bf16_to_f32(q[size_t(wid) * 32 + lane]), code, se);
  qc[size_t(wid) * 32 + lane] = e2m1_int2(code);
  if (lane == 0) qs[wid] = se;
}

constexpr int QG = 4, KT = 128, IH = 64, ID = 128, NT = 256, PAD = 16;
constexpr uint16_t kBf16NegInf = 0xff80;

typedef int int2_t __attribute__((ext_vector_type(2)));
typedef int int8_t_v __attribute__((ext_vector_type(8)));

// Block: QG queries x KT keys; wave w owns keys [16 w, 16 w + 16) of the tile and all 64 heads.
__global__ __launch_bounds__(NT) void indexer_fp4_kernel(const int8_t* __restrict__ qc, const uint8_t* __restrict__ qs,
                                                         const bf16* __restrict__ w, PagedFp4 keys,
                                                         const int32_t* __restrict__ pos, uint16_t* __restrict__ out,
                                                         int T, int nkeys, int ratio) {
  const int k0 = blockIdx.x * KT, t0 = blockIdx.y * QG;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
#if defined(UH_GFX12)
  __shared__ alignas(16) int8_t sA[QG][IH][ID + PAD];
  __shared__ alignas(16) int8_t sK[KT][ID + PAD];
  __shared__ uint8_t sQs[QG][IH][4];
  __shared__ uint8_t sKs[KT][4];
  __shared__ float sW[QG][IH];
  for (int i = tid; i < KT * kFp4RowBytes; i += NT) {
    const int r = i / kFp4RowBytes, c = i % kFp4RowBytes, key = k0 + r;
    const uint32_t byte = key < nkeys ? fp4_codes(keys, key)[c] : 0u;
    sK[r][2 * c] = e2m1_int2(byte & 15);
    sK[r][2 * c + 1] = e2m1_int2(byte >> 4);
  }
  for (int i = tid; i < KT * 4; i += NT) {
    const int r = i / 4, key = k0 + r;
    sKs[r][i % 4] = key < nkeys ? fp4_scales(keys, key)[i % 4] : uint8_t(127);
  }
  for (int i = tid; i < QG * IH * (ID / 16); i += NT) {  // 16-byte pieces
    const int tq = i / (IH * (ID / 16)), rem = i % (IH * (ID / 16)), h = rem / (ID / 16), c = (rem % (ID / 16)) * 16;
    uint4 v = {0, 0, 0, 0};
    if (t0 + tq < T) v = *reinterpret_cast<const uint4*>(qc + (size_t(t0 + tq) * IH + h) * ID + c);
    *reinterpret_cast<uint4*>(&sA[tq][h][c]) = v;
  }
  for (int i = tid; i < QG * IH * 4; i += NT) {
    const int tq = i / (IH * 4);
    sQs[tq][(i / 4) % IH][i % 4] = t0 + tq < T ? qs[size_t(t0) * IH * 4 + i] : uint8_t(127);
  }
  for (int i = tid; i < QG * IH; i += NT) {
    const int tq = i / IH;
    sW[tq][i % IH] = t0 + tq < T ? bf16_to_f32(w[size_t(t0) * IH + i]) : 0.0f;
  }
  __syncthreads();
  const int kc = wave * 16 + (lane & 15);  // this lane's key column within the tile
  const int kofs = (lane >> 4) * 8;        // k offset of this lane's fragment
  for (int tq = 0; tq < QG && t0 + tq < T; ++tq) {
    float facc[4][8];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) facc[r][i] = 0.0f;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      int8_t_v iacc[4];
#pragma unroll
      for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i) iacc[r][i] = 0;
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int kk = b * 32 + s * 16 + kofs;
        const int2_t bfrag = *reinterpret_cast<const int2_t*>(&sK[kc][kk]);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int2_t afrag = *reinterpret_cast<const int2_t*>(&sA[tq][r * 16 + (lane & 15)][kk]);
          iacc[r] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, afrag, true, bfrag, iacc[r], false);
        }
      }
      const int ek = sKs[kc][b];
#pragma unroll
      for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int h = r * 16 + (lane >> 4) * 8 + i;
          facc[r][i] += ldexpf(float(iacc[r][i]), int(sQs[tq][h][b]) + ek - 256);
        }
    }
    // Per (head, key): bf16(relu(bf16(dot)) * w_h); then the head sum in order 0..63. Lane c (< 16) holds
    // heads 16r + 0..7 of key c, lane c + 16 heads 16r + 8..15.
    float total = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      float term[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int h = r * 16 + (lane >> 4) * 8 + i;
        term[i] = round_bf16(fmaxf(round_bf16(facc[r][i]), 0.0f) * sW[tq][h]);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) total += term[i];
#pragma unroll
      for (int i = 0; i < 8; ++i) total += __shfl_down(term[i], 16, 32);
    }
    const int key = k0 + kc, t = t0 + tq;
    if (lane < 16 && key < nkeys) {
      const int visible = (pos[t] + 1) / ratio;
      out[size_t(t) * nkeys + key] = key < visible ? f32_to_bf16(total) : kBf16NegInf;
    }
  }
#else
  // Portable fallback (not used on the devices that run the indexer): one thread per (query, key).
  for (int i = tid; i < QG * KT; i += NT) {
    const int tq = i / KT, key = k0 + i % KT, t = t0 + tq;
    if (t >= T || key >= nkeys) continue;
    const int visible = (pos[t] + 1) / ratio;
    if (key >= visible) {
      out[size_t(t) * nkeys + key] = kBf16NegInf;
      continue;
    }
    const uint8_t* kcodes = fp4_codes(keys, key);
    const uint8_t* ksc = fp4_scales(keys, key);
    float total = 0.0f;
    for (int h = 0; h < IH; ++h) {
      float dot = 0.0f;
      for (int b = 0; b < 4; ++b) {
        int acc = 0;
        for (int j = 0; j < 32; ++j) {
          const int c = b * 32 + j;
          const uint32_t nib = (c & 1) ? kcodes[c / 2] >> 4 : kcodes[c / 2] & 15;
          acc += int(qc[(size_t(t) * IH + h) * ID + c]) * int(e2m1_int2(nib));
        }
        dot += ldexpf(float(acc), int(qs[(size_t(t) * IH + h) * 4 + b]) + int(ksc[b]) - 256);
      }
      total += round_bf16(fmaxf(round_bf16(dot), 0.0f) * bf16_to_f32(w[size_t(t) * IH + h]));
    }
    out[size_t(t) * nkeys + key] = f32_to_bf16(total);
  }
#endif
}

// Order-preserving 16-bit key of a bf16 bit pattern.
__device__ __forceinline__ uint32_t key16(uint16_t b) { return (b & 0x8000u) ? (~uint32_t(b) & 0xffffu) : (b | 0x8000u); }

// One block per row: radix select on the 16-bit key (two 8-bit passes), then collect keys above the
// threshold plus threshold ties by lowest index, and sort (key desc, index asc).
__global__ __launch_bounds__(256) void topk_bf16_kernel(const uint16_t* scores, int n, int k, int32_t* idx) {
  __shared__ uint32_t hist[256];
  __shared__ uint32_t prefix_sel, need_sel, nvalid, count_above;
  __shared__ uint32_t sel_keys[1024];
  __shared__ int32_t sel_idx[1024];
  const uint16_t* row = scores + size_t(blockIdx.x) * n;
  const uint32_t neg_inf_key = key16(kBf16NegInf);
  if (threadIdx.x == 0) nvalid = count_above = 0;
  __syncthreads();
  uint32_t local = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) local += key16(row[i]) > neg_inf_key;
  atomicAdd(&nvalid, local);
  __syncthreads();
  const uint32_t want = min(uint32_t(k), nvalid);
  uint32_t prefix = 0, need = want;
  if (want > 0) {
    for (int pass = 1; pass >= 0; --pass) {
      for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
      __syncthreads();
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t key = key16(row[i]);
        if (key <= neg_inf_key) continue;
        if (pass == 0 && (key >> 8) != (prefix >> 8)) continue;
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
  for (int i = threadIdx.x; i < n && want > 0; i += blockDim.x) {
    const uint32_t key = key16(row[i]);
    if (key > prefix && key > neg_inf_key) {
      const uint32_t slot = atomicAdd(&count_above, 1u);
      sel_keys[slot] = key;
      sel_idx[slot] = i;
    }
  }
  __syncthreads();
  if (threadIdx.x < 32 && want > 0) {  // ties at the threshold, lowest index first
    const uint32_t base = count_above;
    uint32_t taken = 0;
    for (int i0 = 0; i0 < n && taken < need; i0 += 32) {
      const int i = i0 + threadIdx.x;
      const bool tie = i < n && key16(row[i]) == prefix;
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

}  // namespace

void store_rows_fp4(const bf16* rows, int n, PagedFp4 dst, int row0, hipStream_t st) {
  if (n <= 0) return;
  store_rows_fp4_kernel<<<cdiv(size_t(n) * 4 * 32, 256), 256, 0, st>>>(rows, n, dst, row0);
  UH_CHECK_LAUNCH();
}

void load_rows_fp4(PagedFp4 src, int row0, int n, bf16* out, hipStream_t st) {
  if (n <= 0) return;
  load_rows_fp4_kernel<<<cdiv(size_t(n) * 128, 256), 256, 0, st>>>(src, row0, n, out);
  UH_CHECK_LAUNCH();
}

void quant_q_fp4(const bf16* q, int T, int H, int8_t* qc, uint8_t* qs, hipStream_t st) {
  const int blocks = T * H * 4;
  if (blocks <= 0) return;
  quant_q_fp4_kernel<<<cdiv(size_t(blocks) * 32, 256), 256, 0, st>>>(q, blocks, qc, qs);
  UH_CHECK_LAUNCH();
}

void indexer_scores_fp4(const int8_t* qc, const uint8_t* qs, const bf16* w, PagedFp4 keys, const int32_t* pos,
                        uint16_t* scores, int T, int H, int nkeys, int ratio, hipStream_t st) {
  if (H != IH) throw std::runtime_error("indexer_scores_fp4: 64 heads expected");
  if (T <= 0 || nkeys <= 0) return;
  indexer_fp4_kernel<<<dim3(cdiv(size_t(nkeys), KT), cdiv(size_t(T), QG)), NT, 0, st>>>(qc, qs, w, keys, pos, scores,
                                                                                      T, nkeys, ratio);
  UH_CHECK_LAUNCH();
}

void topk_bf16(const uint16_t* scores, int T, int n, int k, int32_t* idx, hipStream_t st) {
  if (k > 1024) throw std::runtime_error("topk_bf16: k > 1024");
  if (T <= 0) return;
  topk_bf16_kernel<<<T, 256, 0, st>>>(scores, n, k, idx);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

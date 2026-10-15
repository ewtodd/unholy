#include "kernels/dev.hip.hpp"
#include <algorithm>

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

// One wave per (query, head): lane holds dims 4l..4l+3, so 32-blocks are 8-lane groups.
__global__ void quant_q_fp4_kernel(const bf16* q, int rows, uint8_t* q8, uint8_t* qe) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (wid >= rows) return;
  const bf16* x = q + size_t(wid) * 128 + lane * 4;
  float v[4];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = bf16_to_f32(x[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
#pragma unroll
  for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  amax = fmaxf(amax, 6.0f * 1.1754943508222875e-38f);
  const uint8_t se = pow2_ceil_e8m0(amax * (1.0f / 6.0f));
  uint32_t emax = se;
  emax = max(emax, uint32_t(__shfl_xor(int(emax), 8, 32)));
  emax = max(emax, uint32_t(__shfl_xor(int(emax), 16, 32)));
  const float sc = e8m0_to_f32(se), rel = ldexpf(1.0f, int(se) - int(emax));
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    packed |= uint32_t(f32_to_e4m3(round_e2m1(fminf(fmaxf(v[i] / sc, -6.0f), 6.0f)) * rel)) << (8 * i);
  reinterpret_cast<uint32_t*>(q8 + size_t(wid) * 128)[lane] = packed;
  if (lane == 0) qe[wid] = uint8_t(emax);
}

// The indexer's query chain in one launch: rope_kernel on each 128-dim head's last 2 * npair dims,
// hadamard128_kernel, quant_q_fp4_kernel. Blocks of 4 waves, a wave per (row, head) segment, the
// Hadamard's LDS butterflies and barriers as hadamard128_kernel, every value through the same operations.
__global__ __launch_bounds__(128) void index_q_chain_kernel(bf16* q, size_t nseg, int heads, const int32_t* pos,
                                                            const float2* freqs, int npair, uint8_t* q8, uint8_t* qe) {
  __shared__ float buf[4][128];
  const int local = threadIdx.x / 32, lane = threadIdx.x & 31;
  const size_t seg = size_t(blockIdx.x) * 4 + local;
  const bool valid = seg < nseg;
  bf16* p = q + seg * 128;
  float* b = buf[local];
  if (valid) {
    for (int i = lane; i < 128; i += 32) b[i] = bf16_to_f32(p[i]);
    if (lane < npair) {  // rope_kernel: its bf16 outputs are what the Hadamard reads
      const int r = int(seg / heads), j = lane, off = 128 - 2 * npair;
      const float a = bf16_to_f32(p[off + 2 * j]), c0 = bf16_to_f32(p[off + 2 * j + 1]);
      const float2 f = freqs[size_t(pos[r]) * npair + j];
      const float c = f.x, s = f.y;
      b[off + 2 * j] = bf16_to_f32(f32_to_bf16(a * c - c0 * s));
      b[off + 2 * j + 1] = bf16_to_f32(f32_to_bf16(a * s + c0 * c));
    }
  }
  __syncthreads();
  for (int h = 1; h < 128; h <<= 1) {
    if (valid)
      for (int t = lane; t < 64; t += 32) {
        const int i = (t / h) * 2 * h + (t % h);
        const float a = b[i], c = b[i + h];
        b[i] = a + c;
        b[i + h] = a - c;
      }
    __syncthreads();
  }
  const float hs = 0.08838834764831845f;  // 128^-0.5
  if (valid)
    for (int i = lane; i < 128; i += 32) {
      const bf16 v = f32_to_bf16(b[i] * hs);
      p[i] = v;
      b[i] = bf16_to_f32(v);
    }
  __syncthreads();
  if (!valid) return;
  // quant_q_fp4_kernel on the segment (lane: values 4 lane .. 4 lane + 3).
  float v[4];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = b[lane * 4 + i];
    amax = fmaxf(amax, fabsf(v[i]));
  }
#pragma unroll
  for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  amax = fmaxf(amax, 6.0f * 1.1754943508222875e-38f);
  const uint8_t se = pow2_ceil_e8m0(amax * (1.0f / 6.0f));
  uint32_t emax = se;
  emax = max(emax, uint32_t(__shfl_xor(int(emax), 8, 32)));
  emax = max(emax, uint32_t(__shfl_xor(int(emax), 16, 32)));
  const float sc = e8m0_to_f32(se), rel = ldexpf(1.0f, int(se) - int(emax));
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    packed |= uint32_t(f32_to_e4m3(round_e2m1(fminf(fmaxf(v[i] / sc, -6.0f), 6.0f)) * rel)) << (8 * i);
  reinterpret_cast<uint32_t*>(q8 + seg * 128)[lane] = packed;
  if (lane == 0) qe[seg] = uint8_t(emax);
}

constexpr int QG = 4, KT = 128, IH = 64, ID = 128, NT = 256, PAD = 16;
constexpr uint16_t kBf16NegInf = 0xff80;

typedef int int2_t __attribute__((ext_vector_type(2)));
typedef float float8_t __attribute__((ext_vector_type(8)));

// Block: QG queries x KT keys; wave w owns keys [16 w, 16 w + 16) of the tile and all 64 heads.
__global__ __launch_bounds__(NT) void indexer_fp4_kernel(const uint8_t* __restrict__ q8, const uint8_t* __restrict__ qe,
                                                         const bf16* __restrict__ w, PagedFp4 keys,
                                                         const int32_t* __restrict__ pos, uint16_t* __restrict__ out,
                                                         int T, int nkeys, int ratio) {
  const int k0 = blockIdx.x * KT, t0 = blockIdx.y * QG;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
#if defined(UH_GFX12)
  __shared__ alignas(16) uint8_t sA[QG][IH][ID + PAD];
  __shared__ alignas(16) uint8_t sK[KT][ID + PAD];
  __shared__ float sKf[KT];       // 2^(key's largest block exponent - 127)
  __shared__ float sWq[QG][IH];   // w_h * 2^(query head's exponent - 127)
  // Keys: e4m3 of e2m1 * 2^(e_b - e_max) through a table [d][nibble] (d = e_max - e_b, 0..8; larger d
  // underflows like f32_to_e4m3 would); thread per (key, 32-block).
  __shared__ uint8_t sLut[9][16];
  if (tid < 9 * 16) sLut[tid / 16][tid % 16] = f32_to_e4m3(e2m1_to_f32(tid % 16) * ldexpf(1.0f, -(tid / 16)));
  __syncthreads();
  for (int i = tid; i < KT * 4; i += NT) {
    const int r = i / 4, b = i % 4, key = k0 + r;
    uint32_t* dst = reinterpret_cast<uint32_t*>(&sK[r][b * 32]);
    float kf = 0.0f;
    if (key < nkeys) {
      const uint8_t* sc = fp4_scales(keys, key);
      const int emax = max(max(sc[0], sc[1]), max(sc[2], sc[3]));
      kf = e8m0_to_f32(uint8_t(emax));
      const uint8_t* lut = sLut[min(emax - int(sc[b]), 8)];
      const uint4 codes = *reinterpret_cast<const uint4*>(fp4_codes(keys, key) + b * 16);
      const uint32_t cw[4] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {  // 4 nibbles -> 4 bytes
        const uint32_t v = cw[j / 2] >> (16 * (j % 2));
        dst[j] = uint32_t(lut[v & 15]) | uint32_t(lut[(v >> 4) & 15]) << 8 | uint32_t(lut[(v >> 8) & 15]) << 16 |
                 uint32_t(lut[(v >> 12) & 15]) << 24;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) dst[j] = 0;
    }
    if (b == 0) sKf[r] = kf;
  }
  for (int i = tid; i < QG * IH * (ID / 16); i += NT) {  // 16-byte pieces
    const int tq = i / (IH * (ID / 16)), rem = i % (IH * (ID / 16)), h = rem / (ID / 16), c = (rem % (ID / 16)) * 16;
    uint4 v = {0, 0, 0, 0};
    if (t0 + tq < T) v = *reinterpret_cast<const uint4*>(q8 + (size_t(t0 + tq) * IH + h) * ID + c);
    *reinterpret_cast<uint4*>(&sA[tq][h][c]) = v;
  }
  for (int i = tid; i < QG * IH; i += NT) {
    const int tq = i / IH;
    sWq[tq][i % IH] = t0 + tq < T ? bf16_to_f32(w[size_t(t0) * IH + i]) * e8m0_to_f32(qe[size_t(t0) * IH + i]) : 0.0f;
  }
  __syncthreads();
  const int kc = wave * 16 + (lane & 15);  // this lane's key column within the tile
  const int kofs = (lane >> 4) * 8;        // k offset of this lane's fragment
  int2_t bfrag[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) bfrag[s] = *reinterpret_cast<const int2_t*>(&sK[kc][s * 16 + kofs]);
  const float kf = sKf[kc];
  for (int tq = 0; tq < QG && t0 + tq < T; ++tq) {
    float8_t acc[4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[r][i] = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int2_t afrag = *reinterpret_cast<const int2_t*>(&sA[tq][r * 16 + (lane & 15)][s * 16 + kofs]);
        acc[r] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(afrag, bfrag[s], acc[r]);
      }
    // dot = acc * 2^(eq - 127) * 2^(ek - 127). Power-of-two factors commute with the bf16 roundings and
    // with fp32 sums, so total = 2^(ek - 127) * sum_h bf16(bf16(relu(acc)) * w_h 2^(eq - 127)), summed in
    // head order as the reference does. Lane c (< 16) holds heads 16r + 0..7 of key c, lane c + 16 heads
    // 16r + 8..15: the running sum is handed to the upper lane and back for every 16 heads.
    float total = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      float term[8];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        term[i] = round_bf16_finite(round_bf16_finite(fmaxf(acc[r][i], 0.0f)) * sWq[tq][r * 16 + (lane >> 4) * 8 + i]);
      float a = total;  // meaningful in lanes < 16
#pragma unroll
      for (int i = 0; i < 8; ++i) a += term[i];
      float b = __shfl_up(a, 16, 32);  // lanes >= 16 continue lane - 16's sum
#pragma unroll
      for (int i = 0; i < 8; ++i) b += term[i];
      total = __shfl_down(b, 16, 32);
    }
    const int key = k0 + kc, t = t0 + tq;
    if (lane < 16 && key < nkeys) {
      const int visible = (pos[t] + 1) / ratio;
      out[size_t(t) * nkeys + key] = key < visible ? f32_to_bf16(total * kf) : kBf16NegInf;
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
      const uint8_t* qr = q8 + (size_t(t) * IH + h) * ID;
      const float qf = e8m0_to_f32(qe[size_t(t) * IH + h]);
      float dot = 0.0f;
      for (int c = 0; c < ID; ++c) {
        const uint32_t nib = (c & 1) ? kcodes[c / 2] >> 4 : kcodes[c / 2] & 15;
        dot += e4m3_to_f32(qr[c]) * qf * e2m1_to_f32(nib) * e8m0_to_f32(ksc[c / 32]);
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
__device__ __forceinline__ void topk_row(const uint16_t* __restrict__ row, int n, int k, int32_t* __restrict__ out) {
  __shared__ uint32_t hist[256];
  __shared__ uint32_t prefix_sel, need_sel, nvalid, count_above;
  __shared__ uint32_t sel_keys[1024];
  __shared__ int32_t sel_idx[1024];
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
  for (int i = threadIdx.x; i < k; i += blockDim.x) out[i] = uint32_t(i) < want ? sel_idx[i] : -1;
}

__global__ __launch_bounds__(256) void topk_bf16_kernel(const uint16_t* scores, int n, int k, int32_t* idx) {
  topk_row(scores + size_t(blockIdx.x) * n, n, k, idx + size_t(blockIdx.x) * k);
}

// Rows of up to TopkBatch::kMax groups (each its own n and k) in one launch: block -> (group, row).
__global__ __launch_bounds__(256) void topk_bf16_batch_kernel(const uint16_t* scores, TopkBatch b, int32_t* idx) {
  int g = 0, first = 0;
  while (g + 1 < b.ngroups && int(blockIdx.x) >= first + b.g[g].rows) first += b.g[g++].rows;
  const TopkRows r = b.g[g];
  const int i = int(blockIdx.x) - first;
  topk_row(scores + r.off + size_t(i) * r.n, r.n, r.k, idx + r.idx_off + size_t(i) * r.k);
}

// ---- Top-k for a few long rows (decode): several workgroups per row --------------------------------
// Exactly topk_row's selection (the same threshold, the lowest-index ties, the same order), in five
// launches of (row, slice) workgroups over TK_SLICE-key slices: high-byte histograms per slice; their
// sum -> the high byte of the threshold, then low-byte histograms of that bin; the threshold and the
// slice's counts above / at it; the candidates (ties ranked by index through the slices' prefix); the
// sort of each row's candidates. Per-row scratch (uint32): hi [TK_MAXS][256], lo [TK_MAXS][256], valid,
// above and tie counts [TK_MAXS] each, state [8], candidate keys and indices [1024] each.
constexpr int TK_SLICE = 2048, TK_MAXS = 128, TK_ROW_WORDS = 2 * TK_MAXS * 256 + 3 * TK_MAXS + 8 + 2 * 1024;

struct TkRow {
  const uint16_t* row;
  int n, k;
  int32_t* out;
  uint32_t *hi, *lo, *valid, *above, *ties, *state, *ckey;
  int32_t* cidx;
};
__device__ __forceinline__ TkRow tk_row(const uint16_t* scores, const TopkBatch& b, int32_t* idx, uint32_t* scr) {
  int g = 0, first = 0;
  while (g + 1 < b.ngroups && int(blockIdx.x) >= first + b.g[g].rows) first += b.g[g++].rows;
  const TopkRows& r = b.g[g];
  const int i = int(blockIdx.x) - first;
  TkRow t;
  t.row = scores + r.off + size_t(i) * r.n;
  t.n = r.n;
  t.k = r.k;
  t.out = idx + r.idx_off + size_t(i) * r.k;
  uint32_t* p = scr + size_t(blockIdx.x) * TK_ROW_WORDS;
  t.hi = p;
  t.lo = p + TK_MAXS * 256;
  t.valid = t.lo + TK_MAXS * 256;
  t.above = t.valid + TK_MAXS;
  t.ties = t.above + TK_MAXS;
  t.state = t.ties + TK_MAXS;  // want, b1, need1, thr, need
  t.ckey = t.state + 8;
  t.cidx = reinterpret_cast<int32_t*>(t.ckey + 1024);
  return t;
}
__host__ __device__ __forceinline__ int tk_slices(int n) { return (n + TK_SLICE - 1) / TK_SLICE; }

// Sum of the slices' histograms into h[256] (LDS), then topk_row's bin search: the highest bin b >= 1
// whose suffix sum reaches `need`, else 0; returns b and sets *above to the count in bins above it.
__device__ int tk_select_bin(const uint32_t* hists, int ns, uint32_t need, uint32_t* h, uint32_t* above) {
  __shared__ uint32_t suf[257];
  __shared__ int sel;
  const int b = threadIdx.x;
  uint32_t v = 0;
  for (int s = 0; s < ns; ++s) v += hists[size_t(s) * 256 + b];
  h[b] = v;
  suf[b] = v;
  if (b == 0) {
    suf[256] = 0;
    sel = 0;
  }
  __syncthreads();
  for (int d = 1; d < 256; d <<= 1) {  // inclusive suffix sums
    const uint32_t add = b + d < 256 ? suf[b + d] : 0;
    __syncthreads();
    suf[b] += add;
    __syncthreads();
  }
  if (b >= 1 && suf[b] >= need && suf[b + 1] < need) sel = b;
  __syncthreads();
  *above = suf[sel + 1];
  return sel;
}

__global__ __launch_bounds__(256) void tk_hi_kernel(const uint16_t* scores, TopkBatch bt, int32_t* idx, uint32_t* scr) {
  const TkRow r = tk_row(scores, bt, idx, scr);
  const int s = blockIdx.y;
  if (s >= tk_slices(r.n)) return;
  __shared__ uint32_t hist[256], nv;
  hist[threadIdx.x] = 0;
  if (threadIdx.x == 0) nv = 0;
  __syncthreads();
  const uint32_t neg = key16(kBf16NegInf);
  uint32_t c = 0;
  for (int i = s * TK_SLICE + threadIdx.x; i < min(r.n, (s + 1) * TK_SLICE); i += 256) {
    const uint32_t key = key16(r.row[i]);
    if (key > neg) {
      atomicAdd(&hist[key >> 8], 1u);
      ++c;
    }
  }
  atomicAdd(&nv, c);
  __syncthreads();
  r.hi[size_t(s) * 256 + threadIdx.x] = hist[threadIdx.x];
  if (threadIdx.x == 0) r.valid[s] = nv;
}

__global__ __launch_bounds__(256) void tk_lo_kernel(const uint16_t* scores, TopkBatch bt, int32_t* idx, uint32_t* scr) {
  const TkRow r = tk_row(scores, bt, idx, scr);
  const int s = blockIdx.y, ns = tk_slices(r.n);
  if (s >= ns) return;
  __shared__ uint32_t h[256], hist[256], nvalid;
  if (threadIdx.x == 0) nvalid = 0;
  __syncthreads();
  if (threadIdx.x < ns) atomicAdd(&nvalid, r.valid[threadIdx.x]);
  __syncthreads();
  const uint32_t want = min(uint32_t(r.k), nvalid);
  uint32_t above = 0;
  const int b1 = tk_select_bin(r.hi, ns, want, h, &above);
  if (s == 0 && threadIdx.x == 0) {
    r.state[0] = want;
    r.state[1] = uint32_t(b1);
    r.state[2] = want - above;
  }
  hist[threadIdx.x] = 0;
  __syncthreads();
  const uint32_t neg = key16(kBf16NegInf);
  if (want > 0)
    for (int i = s * TK_SLICE + threadIdx.x; i < min(r.n, (s + 1) * TK_SLICE); i += 256) {
      const uint32_t key = key16(r.row[i]);
      if (key > neg && int(key >> 8) == b1) atomicAdd(&hist[key & 0xff], 1u);
    }
  __syncthreads();
  r.lo[size_t(s) * 256 + threadIdx.x] = hist[threadIdx.x];
}

__global__ __launch_bounds__(256) void tk_count_kernel(const uint16_t* scores, TopkBatch bt, int32_t* idx,
                                                      uint32_t* scr) {
  const TkRow r = tk_row(scores, bt, idx, scr);
  const int s = blockIdx.y, ns = tk_slices(r.n);
  if (s >= ns) return;
  __shared__ uint32_t h[256], na, nt;
  const uint32_t want = r.state[0], b1 = r.state[1], need1 = r.state[2];
  uint32_t above = 0;
  const int b0 = tk_select_bin(r.lo, ns, need1, h, &above);
  const uint32_t thr = (b1 << 8) | uint32_t(b0), need = need1 - above;
  if (s == 0 && threadIdx.x == 0) {
    r.state[3] = thr;
    r.state[4] = need;
  }
  if (threadIdx.x == 0) na = nt = 0;
  __syncthreads();
  const uint32_t neg = key16(kBf16NegInf);
  uint32_t a = 0, t = 0;
  if (want > 0)
    for (int i = s * TK_SLICE + threadIdx.x; i < min(r.n, (s + 1) * TK_SLICE); i += 256) {
      const uint32_t key = key16(r.row[i]);
      a += key > thr && key > neg;
      t += key == thr;
    }
  atomicAdd(&na, a);
  atomicAdd(&nt, t);
  __syncthreads();
  if (threadIdx.x == 0) {
    r.above[s] = na;
    r.ties[s] = nt;
  }
}

__global__ __launch_bounds__(256) void tk_emit_kernel(const uint16_t* scores, TopkBatch bt, int32_t* idx, uint32_t* scr) {
  const TkRow r = tk_row(scores, bt, idx, scr);
  const int s = blockIdx.y, ns = tk_slices(r.n);
  if (s >= ns) return;
  const uint32_t want = r.state[0], thr = r.state[3], need = r.state[4];
  if (want == 0) return;
  __shared__ uint32_t pa, pt, cnt, wties[8], wpre[8];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    uint32_t x = 0, y = 0;
    for (int j = 0; j < s; ++j) {
      x += r.above[j];
      y += r.ties[j];
    }
    pa = x;
    pt = y;
    cnt = 0;
  }
  __syncthreads();
  const uint32_t neg = key16(kBf16NegInf);
  const int i0 = s * TK_SLICE, i1 = min(r.n, (s + 1) * TK_SLICE);
  for (int i = i0 + threadIdx.x; i < i1; i += 256) {
    const uint32_t key = key16(r.row[i]);
    if (key > thr && key > neg) {
      const uint32_t p = pa + atomicAdd(&cnt, 1u);
      r.ckey[p] = key;
      r.cidx[p] = i;
    }
  }
  if (pt >= need) return;
  // Ties in index order: wave w takes keys [i0 + w * 256, i0 + (w + 1) * 256) in 32-key steps.
  const int w0 = i0 + wave * 256, w1 = min(i1, w0 + 256);
  uint32_t c = 0;
  for (int b = w0; b < w1; b += 32) c += __popcll(__ballot(b + lane < w1 && key16(r.row[b + lane]) == thr));
  if (lane == 0) wties[wave] = c;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t x = 0;
    for (int w = 0; w < 8; ++w) {
      wpre[w] = x;
      x += wties[w];
    }
  }
  __syncthreads();
  const uint32_t base = want - need;  // the candidates above the threshold come first
  uint32_t rank = pt + wpre[wave];
  for (int b = w0; b < w1 && rank < need; b += 32) {
    const int i = b + lane;
    const bool tie = i < w1 && key16(r.row[i]) == thr;
    const uint64_t m = __ballot(tie);
    const uint32_t mine = rank + __popcll(m & ((1ull << lane) - 1));
    if (tie && mine < need) {
      r.ckey[base + mine] = thr;
      r.cidx[base + mine] = i;
    }
    rank += __popcll(m);
  }
}

__global__ __launch_bounds__(256) void tk_sort_kernel(const uint16_t* scores, TopkBatch bt, int32_t* idx, uint32_t* scr) {
  const TkRow r = tk_row(scores, bt, idx, scr);
  __shared__ uint32_t sel_keys[1024];
  __shared__ int32_t sel_idx[1024];
  const uint32_t want = r.state[0];
  const int k = r.k;
  int size = 1;
  while (size < k) size <<= 1;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    const bool in = uint32_t(i) < want;
    sel_keys[i] = in ? r.ckey[i] : 0;
    sel_idx[i] = in ? r.cidx[i] : 0x7fffffff;
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
  for (int i = threadIdx.x; i < k; i += blockDim.x) r.out[i] = uint32_t(i) < want ? sel_idx[i] : -1;
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

void quant_q_fp4(const bf16* q, int T, int H, uint8_t* q8, uint8_t* qe, hipStream_t st) {
  const int rows = T * H;
  if (rows <= 0) return;
  quant_q_fp4_kernel<<<cdiv(size_t(rows) * 32, 256), 256, 0, st>>>(q, rows, q8, qe);
  UH_CHECK_LAUNCH();
}

void index_q_chain(bf16* q, int T, int H, const int32_t* pos, const float2* freqs, int npair, uint8_t* q8, uint8_t* qe,
                   hipStream_t st) {
  const size_t nseg = size_t(T) * H;
  if (nseg == 0) return;
  if (npair > 32) throw std::runtime_error("index_q_chain: npair > 32");
  index_q_chain_kernel<<<cdiv(nseg, 4), 128, 0, st>>>(q, nseg, H, pos, freqs, npair, q8, qe);
  UH_CHECK_LAUNCH();
}

void indexer_scores_fp4(const uint8_t* q8, const uint8_t* qe, const bf16* w, PagedFp4 keys, const int32_t* pos,
                        uint16_t* scores, int T, int H, int nkeys, int ratio, hipStream_t st) {
  if (H != IH) throw std::runtime_error("indexer_scores_fp4: 64 heads expected");
  if (T <= 0 || nkeys <= 0) return;
  indexer_fp4_kernel<<<dim3(cdiv(size_t(nkeys), KT), cdiv(size_t(T), QG)), NT, 0, st>>>(q8, qe, w, keys, pos, scores,
                                                                                      T, nkeys, ratio);
  UH_CHECK_LAUNCH();
}

size_t topk_split_scratch_words(int rows) { return size_t(rows) * TK_ROW_WORDS; }

void topk_bf16_batch(const uint16_t* scores, const TopkBatch& b, int32_t* idx, hipStream_t st, uint32_t* scratch,
                     size_t scratch_words) {
  if (b.ngroups <= 0 || b.ngroups > TopkBatch::kMax) throw std::runtime_error("topk_bf16_batch: group count");
  int rows = 0, maxn = 0;
  for (int i = 0; i < b.ngroups; ++i) {
    if (b.g[i].k > 1024) throw std::runtime_error("topk_bf16_batch: k > 1024");
    rows += b.g[i].rows;
    maxn = std::max(maxn, b.g[i].n);
  }
  if (rows == 0) return;
  // A few long rows (decode): several workgroups per row; otherwise one per row.
  if (scratch && maxn >= kTopkSplitMinKeys && maxn <= TK_SLICE * TK_MAXS && rows <= kTopkSplitMaxRows &&
      topk_split_scratch_words(rows) <= scratch_words) {
    const dim3 grid(rows, tk_slices(maxn));
    tk_hi_kernel<<<grid, 256, 0, st>>>(scores, b, idx, scratch);
    tk_lo_kernel<<<grid, 256, 0, st>>>(scores, b, idx, scratch);
    tk_count_kernel<<<grid, 256, 0, st>>>(scores, b, idx, scratch);
    tk_emit_kernel<<<grid, 256, 0, st>>>(scores, b, idx, scratch);
    tk_sort_kernel<<<rows, 256, 0, st>>>(scores, b, idx, scratch);
    UH_CHECK_LAUNCH();
    return;
  }
  topk_bf16_batch_kernel<<<rows, 256, 0, st>>>(scores, b, idx);
  UH_CHECK_LAUNCH();
}

void topk_bf16(const uint16_t* scores, int T, int n, int k, int32_t* idx, hipStream_t st) {
  if (k > 1024) throw std::runtime_error("topk_bf16: k > 1024");
  if (T <= 0) return;
  topk_bf16_kernel<<<T, 256, 0, st>>>(scores, n, k, idx);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

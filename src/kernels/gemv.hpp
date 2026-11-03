// Skinny GEMMs for M <= 32 (decode). Same contracts as the WMMA GEMMs (gemm_wmma.hpp); K % 512 == 0.
// A row's result does not depend on M.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/elem_fast.hpp"

namespace unholy::kernels {

using bf16 = uint16_t;

// Dense layers use the GEMVs for steps of up to gemv_max_rows() rows: UNHOLY_GEMV_ROWS, default 8 (the
// 2026-10-12 build's threshold, so every step's rows come out as before). 16 is faster for 9-16-row steps
// (several streams' verify chunks; unholy-gemv-test) and gives those rows the single-stream values, but
// changes them against the default. The kernels take up to kGemvRowsCap rows.
constexpr int kGemvRowsCap = 32;
int gemv_max_rows();

// hp.x set: also the mHC post-mix of each output column into the rows' streams (hc_post_slices' arithmetic; N = dim).
void gemv_w8a8(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc, int M,
               int N, int K, hipStream_t st, const HcPostArgs& hp = {});
// gemv_bf16 also takes K < 512 with K % 64 == 0. scale != 0 (K >= 512): the outputs scaled as scale_bf16 does.
void gemv_bf16(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st,
               float scale = 0.0f);
void gemv_bf16_f32out(const bf16* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K, hipStream_t st);
// act_quant_fp8 (block 128) of a GEMV's outputs in its launch: each 128-column segment's last block quantizes its
// rows into q [M][N] / qs [M][N / 128] (N: all outputs). counters [N / 128]: zero, left zero.
struct GemvQuant {
  uint8_t* q = nullptr;
  uint8_t* qs = nullptr;
  uint32_t* counters = nullptr;
};
void gemv_grouped_fp8w(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                       hipStream_t st, const GemvQuant& gq = {});
// Up to three GEMVs on one input in one launch: part i is (weights, scales (W8A8 only), output, ldc, N);
// every column is computed exactly as by the single GEMV.
struct GemvPart {
  const void* w;
  const uint8_t* sw;
  void* c;
  int ldc, N;
};
struct GemvParts {
  static constexpr int kMax = 3;
  int n = 0;
  GemvPart p[kMax];
};
void gemv_w8a8_parts(const uint8_t* a, const uint8_t* sa, const GemvPart* parts, int nparts, int M, int K,
                     hipStream_t st);
// The two parts gate / up (one width I) and, by each 128-column segment's last block, swiglu (kernels::swiglu's
// arithmetic, no weights) and act_quant_fp8 of its products into gq (q / qs not the input's codes: blocks of other
// segments are still reading those).
void gemv_w8a8_glu(const uint8_t* a, const uint8_t* sa, const GemvPart* parts, int M, int K, float limit,
                   const GemvQuant& gq, hipStream_t st);
// A compressor's projections (two parts of width N: kv, score) with add_ape and ring_update in the stores: part 1's
// values += ape[pos[m] % ratio], and rows with slot[m] >= 0 also store both values into that slot's ring row
// pos[m] % ring.
struct GemvRing {
  const float* ape = nullptr;  // [ratio][N]
  const int32_t* pos = nullptr;
  int ratio = 0, ring = 0;
  float* kv = nullptr;  // rings [slots][ring][N]
  float* sc = nullptr;
  int slot[kGemvRowsCap];
};
void gemv_bf16_f32out_parts(const bf16* a, int lda, const GemvPart* parts, int nparts, int M, int K, hipStream_t st,
                            const GemvRing* ring = nullptr);
void gemv_bf16_parts(const bf16* a, int lda, const GemvPart* parts, int nparts, int M, int K, hipStream_t st);
// Two of gemv_bf16's K < 512 GEMVs, each with its own input, in one launch (the same values as two launches).
void gemv_bf16_smallk2(const bf16* a0, int lda0, const bf16* w0, bf16* c0, int ldc0, int N0, const bf16* a1, int lda1,
                       const bf16* w1, bf16* c1, int ldc1, int N1, int M, int K, hipStream_t st);
// Tests: force the columns per wave (1, 2 or 4; 0 = by shape). Results do not depend on it.
void gemv_force_cols(int c);

}  // namespace unholy::kernels

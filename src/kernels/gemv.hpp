// Skinny GEMMs for M <= 32 (decode). Same contracts as the WMMA GEMMs (gemm_wmma.hpp); K % 512 == 0.
// A row's result does not depend on M.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace unholy::kernels {

using bf16 = uint16_t;

// Dense layers use the GEMVs for steps of up to gemv_max_rows() rows: UNHOLY_GEMV_ROWS, default 8 (the
// 2026-10-12 build's threshold, so every step's rows come out as before). 16 is faster for 9-16-row steps
// (several streams' verify chunks; unholy-gemv-test) and gives those rows the single-stream values, but
// changes them against the default. The kernels take up to kGemvRowsCap rows.
constexpr int kGemvRowsCap = 32;
int gemv_max_rows();

void gemv_w8a8(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc, int M,
               int N, int K, hipStream_t st);
// gemv_bf16 also takes K < 512 with K % 64 == 0.
void gemv_bf16(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st);
void gemv_bf16_f32out(const bf16* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K, hipStream_t st);
void gemv_grouped_fp8w(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                       hipStream_t st);
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
void gemv_bf16_f32out_parts(const bf16* a, int lda, const GemvPart* parts, int nparts, int M, int K, hipStream_t st);
// Tests: force the columns per wave (1, 2 or 4; 0 = by shape). Results do not depend on it.
void gemv_force_cols(int c);

}  // namespace unholy::kernels

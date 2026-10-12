// Skinny GEMMs for M <= 8 (decode). Same contracts as the WMMA GEMMs (gemm_wmma.hpp); K % 512 == 0.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace unholy::kernels {

using bf16 = uint16_t;

constexpr int kGemvMaxRows = 8;

void gemv_w8a8(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc, int M,
               int N, int K, hipStream_t st);
void gemv_bf16(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st);
void gemv_bf16_f32out(const bf16* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K, hipStream_t st);
void gemv_grouped_fp8w(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                       hipStream_t st);

}  // namespace unholy::kernels

// WMMA GEMMs (fast path). C = A * W^T, W row-major [N, K]. See gemm_wmma.hip.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace unholy::kernels {

using bf16 = uint16_t;

// gfx12 only. a e4m3 [M, K], sa e8m0 [M, K/128]; w e4m3 [N, K], sw e8m0 [ceil(N/128), K/128].
// M <= 64: 16-column tiles with the K blocks spread over the waves (bit-identical; UNHOLY_GEMM_SKINNY=0
// takes the 64 x 128 tiles).
void gemm_w8a8_wmma(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc,
                    int M, int N, int K, hipStream_t st);
// Tests: 1 / 0 forces the skinny / tiled kernel for M <= 64, -1 restores the default.
void gemm_force_skinny(int on);
// gfx11 + gfx12. bf16 x bf16 -> bf16 / fp32.
void gemm_bf16_wmma(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st);
void gemm_bf16_f32out_wmma(const bf16* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K,
                           hipStream_t st);
// Grouped "mgk,grk->mgr" with fp8 weights x 128x128 scales (dequantized to bf16, exact).
void gemm_grouped_fp8w_wmma(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                            hipStream_t st);

}  // namespace unholy::kernels

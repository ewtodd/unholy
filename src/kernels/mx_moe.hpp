// Native MXFP4 routed-expert kernels (fast path; reference: gemm_a8w4 + act_quant_fp8 in ref.hpp).
//
// The checkpoint's experts are e2m1 weights with one e8m0 scale per 32 inputs ([N, K/2] packed + [N,
// K/32]); the reference multiplies them with activations quantized to e4m3 per 128 (e8m0 scales),
// summing each 32-block exactly and scaling it. Both operands dequantize exactly to bf16 (an e4m3 or
// e2m1 value times a power of two), so these kernels fake-quantize the activations to bf16, dequantize
// each weight slice in the wave that consumes it and use bf16 WMMA with fp32 accumulation: the same
// products, summed in a different fp32 order. Tile lists and pointer tables as the EXL3 path
// (exl3_moe.hpp): tables [w1, w3, w2] x [weights, scales] x E, null where an expert is not resident.
//
// Weights are repacked at load into a tile-major layout (lossless): for each 64-input slab kt and
// 16-output tile nt, chunk (kt * N/16 + nt) holds 512 bytes of codes — output n's 32 bytes (inputs
// 64 kt .. +63, two per byte, low nibble first) at (n % 16) * 32 — and 32 bytes of scales — n's two
// 32-blocks at (n % 16) * 2. A wave's slab is then one contiguous 512-byte read.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/exl3_moe.hpp"
#include "kernels/moe_dispatch.hpp"

namespace unholy::kernels {

// [N, K/2] codes + [N, K/32] scales -> the tile-major layout, in place (tmp: N*K/2 + N*K/32 bytes).
void mx_repack(uint8_t* w, uint8_t* s, int N, int K, uint8_t* tmp, hipStream_t st);

// xa[r] = bf16(fq8_128(x[r])) for rows whose expert is resident (scale table entry non-null).
// x_rows: pair row r's input is x row x_rows[r] (the token) instead of row r (gathered pair rows).
void mx_xform_in(const bf16* x, int ldx, const int32_t* row_expert, const uint8_t* const* s_tab, bf16* xa, int P,
                 int K, hipStream_t st, const int32_t* x_rows = nullptr);
// hh[r] = bf16(fq8_128(bf16(silu(min(g, lim)) * clamp(u, +-lim) * w[r]))).
// The same xa rows rebuilt from compact fp8 inputs (quant_remote_tokens): row r in [plan->remote0,
// plan->P) whose expert is resident takes token p2u[r]'s codes and scale exponents. Bit-identical to
// mx_xform_in on the pair rows those tokens would have been gathered into.
void mx_xform_in_q(const uint8_t* q, const uint8_t* s, const int32_t* p2u, const MoePlan* plan,
                   const int32_t* row_expert, const uint8_t* const* s_tab, bf16* xa, int P, int K, hipStream_t st);
void mx_swiglu(const bf16* g, const bf16* u, const float* w, const int32_t* row_expert, const uint8_t* const* s_tab,
               bf16* hh, int P, int I, float limit, hipStream_t st);
// out[r] = xa[r] @ W[e]^T for the rows of each tile (<= kExl3TileRows); K % 64 == 0, N % 128 == 0.
// With w_tab2 / s_tab2 / out2: a second projection of the same rows (w1 and w3) in the same launch.
void mx_grouped_gemm(const bf16* xa, int K, const Exl3Tile* tiles, int ntiles, const uint8_t* const* w_tab,
                     const uint8_t* const* s_tab, bf16* out, int ldo, int N, hipStream_t st,
                     const int* ntiles_dev = nullptr, const uint8_t* const* w_tab2 = nullptr,
                     const uint8_t* const* s_tab2 = nullptr, bf16* out2 = nullptr);
// The same for tiles of <= kMxGemvTileRows rows (decode): a wave per output column, K across the lanes.
// Each row's result does not depend on the tile it is in, so steps of up to mx_gemv_max_tokens() tokens
// plan their experts in tiles of kMxGemvTileRows rows and every row gets what a one-token step gives it.
constexpr int kMxGemvTileRows = 8;
// UNHOLY_MX_GEMV_TOKENS: default 8 (the 2026-10-12 build's threshold: every step's rows as before); 64 is
// faster for several streams' verify steps and gives their rows the single-stream values.
int mx_gemv_max_tokens();
inline bool mx_use_gemv(int tokens) { return tokens <= mx_gemv_max_tokens(); }
void mx_grouped_gemv(const bf16* xa, int K, const Exl3Tile* tiles, int ntiles, const uint8_t* const* w_tab,
                     const uint8_t* const* s_tab, bf16* out, int ldo, int N, hipStream_t st,
                     const int* ntiles_dev = nullptr, const uint8_t* const* w_tab2 = nullptr,
                     const uint8_t* const* s_tab2 = nullptr, bf16* out2 = nullptr);

}  // namespace unholy::kernels

// EXL3 grouped MoE kernels (fast path; reference: kernels/ref_exl3.hip). fp16 storage is uint16_t.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace unholy::kernels {

using bf16 = uint16_t;

// A run of up to kExl3TileRows pair rows of one expert.
constexpr int kExl3TileRows = 128;
struct Exl3Tile {
  int expert, row0, rows;
};

// xh[r] = fp16(H128(fq(x[r]) * suh[e_r]) / sqrt(128)); fq = fp8 128-block fake quant if enabled. Rows whose expert
// has no suh are skipped; with row0_dev / row1_dev (device ints, e.g. a MoePlan's remote0), only rows
// [*row0_dev, *row1_dev) are visited (default 0 / P). With x_rows, row r reads x row x_rows[r] (compact token rows).
void exl3_xform_in(const bf16* x, int ldx, const int32_t* row_expert, const uint16_t* const* suh_tab, uint16_t* xh,
                   int P, int K, bool fake_quant, hipStream_t st, const int* row0_dev = nullptr,
                   const int* row1_dev = nullptr, const int32_t* x_rows = nullptr);
// xh[r] = fp16(H128(fq(bf16(silu(min(g,lim)) * clamp(u,±lim) * w[r])) * suh[e_r]) / sqrt(128)); fq = fp8 128-block
// fake quant unless fake_quant is false; w may be null (no routing weight). Rows as exl3_xform_in's.
void exl3_swiglu_xform(const bf16* g, const bf16* u, const float* w, const int32_t* row_expert,
                       const uint16_t* const* suh_tab, uint16_t* xh, int P, int I, float limit, hipStream_t st,
                       bool fake_quant = true, const int* row0_dev = nullptr, const int* row1_dev = nullptr);
// out[r] = bf16(H128(xh[r] @ W_inner[e]) * svh[e] / sqrt(128)) for the rows of each tile. With
// ntiles_dev, the grid is sized for `ntiles` and only the first *ntiles_dev tiles run. Tiles of up to
// kExl3TileRows rows; K % 64 == 0, N % 128 == 0.
void exl3_grouped_gemm(const uint16_t* xh, int K, const Exl3Tile* tiles, int ntiles,
                       const uint16_t* const* trellis_tab, const uint16_t* const* svh_tab, int bits, bf16* out,
                       int ldo, int N, hipStream_t st, const int* ntiles_dev = nullptr);
// The previous kernel (tiles of <= 64 rows, weights decoded per 64-row tile), kept for comparison.
void exl3_grouped_gemm_v1(const uint16_t* xh, int K, const Exl3Tile* tiles, int ntiles,
                          const uint16_t* const* trellis_tab, const uint16_t* const* svh_tab, int bits, bf16* out,
                          int ldo, int N, hipStream_t st, const int* ntiles_dev = nullptr);

// Decode variant for tiles of <= 8 rows (batches of <= 8 tokens): K split across waves, each lane
// decoding 8 trellis steps of a tile in registers; y' (fp32 [P, N]) then the output transform.
// counters (ntiles * N / 128 zeroed words, left zeroed; one set per stream): the output transform in the same launch,
// by the last of each 128-column segment's 8 blocks to finish (bit for bit).
void exl3_grouped_gemv(const uint16_t* xh, int K, const Exl3Tile* tiles, int ntiles, const uint16_t* const* trellis_tab,
                       const uint16_t* const* svh_tab, int bits, float* yp, bf16* out, int ldo, int N, hipStream_t st,
                       const int* ntiles_dev = nullptr, uint32_t* counters = nullptr);
// A dense linear (tiles of <= 8 rows, one weight set) in one launch: bit for bit exl3_xform_in (no fake quant) +
// exl3_grouped_gemv with counters, each block transforming its tile's input rows x [rows][ldx] (ldx % 4 == 0) itself.
// For one tile of a few rows the redundant transforms cost less than the launch; for experts' many tiles they do not.
void exl3_linear_gemv(const bf16* x, int ldx, int K, const Exl3Tile* tiles, int ntiles, const uint16_t* const* tab,
                      int bits, float* yp, uint32_t* counters, bf16* out, int ldo, int N, hipStream_t st);

}  // namespace unholy::kernels

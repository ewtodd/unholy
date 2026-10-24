// GLM-5.3-Flash fast kernels for DSA layers over the paged cache (model/glm5.hpp "fast path"; references in
// kernels/ref_glm.hpp).
//
// Paged layout, per DSA layer, kPageRows = 256 positions per page (the scheduler's page): the MLA latent row of
// every position in e4m3 (512 codes) plus one power-of-two scale per 64 values (e8m0, 8 per row); the indexer's
// pooled keys (bf16, 128 values) of every complete 4-token pool (64 per page). A slot's block table maps its
// logical pages to physical ones. The keys and gate scores of a slot's incomplete pool (<= 3 positions) are carried
// in a per-slot tail, not paged.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace unholy::kernels {

using bf16 = uint16_t;

constexpr int kGlmPageRows = 256;

struct GlmPaged {
  uint8_t* lat = nullptr;      // e4m3 [pages][256][512]
  uint8_t* lat_s = nullptr;    // e8m0 [pages][256][8]
  bf16* pk = nullptr;          // bf16 [pages][64][128]
  const int32_t* table = nullptr;  // this slot's row of the block table
};

// The n latent rows (bf16 [n][512]) of positions [s0, s0 + n) into the slot's pages: fake_quant_fp8's arithmetic
// (block 64, amax floored at 1e-4, power-of-two scale).
void glm_store_latent(const bf16* lat, GlmPaged pg, int s0, int n, hipStream_t st);

// Pools completed by positions [s0, s0 + n) of one sequence: keys k / gate scores g of the chunk's rows (bf16 [n][128]),
// the slot's tail (k, g: bf16 [4][128], positions s0 - s0 % 4 ..  s0 - 1 at index pos % 4) for the pool the chunk
// starts inside; kernels::idx_pool's arithmetic (ape f32 [4][128]); pooled keys into the slot's pages. Then the
// tail takes the chunk's last (s0 + n) % 4 positions.
void glm_pool_keys(const bf16* k, const bf16* g, bf16* tail_k, bf16* tail_g, const float* ape, GlmPaged pg, int s0,
                   int n, hipStream_t st);

// Index scores (kernels::idx_scores' math, WMMA on gfx12): s[t][p] = sum_h w'[t][h] relu(scale q[t][h] . pk[p]) for
// pools p < npool of the slot's pages, -inf where pool p is not complete by pos[t]. q bf16 [n][32][128], w bf16
// [n][32] (w' = w / sqrt(32)); out f32 [n][ld].
void glm_idx_scores(const bf16* q, const bf16* w, GlmPaged pg, const int32_t* pos, float* out, int ld, int n,
                    int npool, float scale, hipStream_t st);

// Absorbed NoPE MLA (kernels::mla_attn's math with bf16 queries and bf16-rounded probabilities, WMMA on gfx12): one
// block per query, all 64 heads over the fp8 latent rows of positions idx[t][0 .. w) (-1 skipped) of the slot's
// pages. qa / o bf16 [n][64][512].
void glm_mla_attn(const bf16* qa, const int32_t* idx, int w, GlmPaged pg, bf16* o, int n, float scale, hipStream_t st);

// Grouped bf16 GEMM over heads (gemm_wmma's bf16 kernel with groups): c [M, G, R] = a [M, G, K] . w[g]^T, w [G][R][K].
void gemm_grouped_bf16_wmma(const bf16* a, const bf16* w, bf16* c, int M, int G, int R, int K, hipStream_t st);

}  // namespace unholy::kernels

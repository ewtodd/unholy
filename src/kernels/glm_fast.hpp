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

// Pools completed by positions [s0, s0 + n) of one sequence: keys k / gate scores g of the chunk's rows (bf16
// [n][128]), the slot's tail (k, g: bf16 [4][128], positions s0 - s0 % 4 .. s0 - 1 at index pos % 4) for the pool the
// chunk starts inside; kernels::idx_pool's arithmetic (ape f32 [4][128]); pooled keys into the slot's pages. Then the
// tail takes the chunk's last <= 4 positions (also when they complete a pool: a rejected draft can reopen it).
void glm_pool_keys(const bf16* k, const bf16* g, bf16* tail_k, bf16* tail_g, const float* ape, GlmPaged pg, int s0,
                   int n, hipStream_t st);

// Index scores (kernels::idx_scores' math, WMMA on gfx12): s[t][p] = sum_h w'[t][h] relu(scale q[t][h] . pk[p]) for
// pools p < npool of the slot's pages, -inf where pool p is not complete by pos[t]. q bf16 [n][32][128], w bf16
// [n][32] (w' = w / sqrt(32)); out f32 [n][ld].
void glm_idx_scores(const bf16* q, const bf16* w, GlmPaged pg, const int32_t* pos, float* out, int ld, int n,
                    int npool, float scale, hipStream_t st);

// Absorbed NoPE MLA (kernels::mla_attn's math with bf16 queries and bf16-rounded probabilities, WMMA on gfx12): one
// block per query, all 64 heads over the fp8 latent rows of positions idx[t][0 .. glm_sel_count(pos[t])) (a compact
// list, glm_idx_expand; -1 skipped) of the slot's pages. qa / o bf16 [n][64][512].
void glm_mla_attn(const bf16* qa, const int32_t* idx, int w, const int32_t* pos, GlmPaged pg, bf16* o, int n,
                  float scale, hipStream_t st);

// The sigmoid router (kernels::route_sigmoid's selection and weights), one block per token.
void glm_route(const float* logits, const float* bias, int32_t* sel, float* weights, int T, int E, int K, float scale,
               hipStream_t st);

// KDA delta rule (kernels::kda_recurrent's math) in two passes: glm_kda_prep normalizes each token's q and k per head
// (q also scaled by D^-0.5) and takes exp of the forget gate, all fp32 [T][H][D]; glm_kda_recurrent then runs every
// chunk's tokens in order, one block per (head, chunk), 16 tokens per LDS tile. v: bf16 rows of H * D (stride ldv),
// beta bf16 [T][H], state f32 [slot][H][D][D] (k index major), o bf16 [T][H][D].
struct GlmKdaChunk {
  int t0, n, slot;
  int save_after = -1;  // >= 0: also write the state as it is after token save_after of the chunk to the backup
};
struct GlmKdaBatch {
  static constexpr int kMax = 64;
  int nchunks = 0;
  GlmKdaChunk c[kMax];
};
void glm_kda_prep(const bf16* q, const bf16* k, int ld, const float* g, float* qn, float* kn, float* eg, int T, int H,
                  int D, hipStream_t st);
void glm_kda_recurrent(const float* qn, const float* kn, const float* eg, const bf16* v, int ldv, const bf16* beta,
                       float* state, bf16* o, const GlmKdaBatch& b, int H, int D, hipStream_t st,
                       float* backup = nullptr);
// kernels::kda_conv over every chunk of the batch in one launch: x [T, C] (stride ldx), state [slot][3][C], y [T, C];
// chunks with save_after >= 0 also write their state after that token to backup [slot][3][C].
void glm_kda_conv(const bf16* x, int ldx, bf16* state, const bf16* w, bf16* y, const GlmKdaBatch& b, int C,
                  hipStream_t st, bf16* backup = nullptr);

// x = bf16(x + y) (plain residual adds of the MTP layer).
void glm_add_bf16(bf16* x, const bf16* y, size_t n, hipStream_t st);
// The MTP layer's input: cat [rows][2 dim] = [bf16(rms(emb) * enorm), bf16(rms(h) * hnorm)] (fp32 inside).
void glm_mtp_cat(const bf16* emb, const bf16* h, const bf16* enorm, const bf16* hnorm, bf16* cat, int rows, int dim,
                 float eps, hipStream_t st);
// Per row of logits [rows][V]: the argmax (ties: lower id) and its softmax probability.
void glm_argmax_conf(const float* logits, int rows, int V, int32_t* tok, float* conf, hipStream_t st);

// Top-k (desc, ties: lower index, -1 padded) of each of the n rows of scores [n][ld] over cols [0, cols): one block per
// row when cols <= kGlmTopkSeg, else two stages (each 4096-column segment's top k, then their candidates') through
// cand ([n][ceil(cols / seg)][k] values and indices; glm_topk_cand_bytes). k <= 1024. idx [n][k].
constexpr int kGlmTopkSeg = 4096;
size_t glm_topk_cand_bytes(int rows, int cols, int k);
void glm_topk(const float* scores, int ld, int n, int cols, int k, int32_t* idx, void* cand, hipStream_t st);
// glm_mla_attn for decode-sized calls: the keys split over `splits` workgroups per query (partial softmax states in
// part: glm_mla_part_bytes), then combined. The same keys and arithmetic per split as the fused kernel's blocks.
size_t glm_mla_part_bytes(int n, int splits);
void glm_mla_attn_split(const bf16* qa, const int32_t* idx, int w, const int32_t* pos, GlmPaged pg, bf16* o, int n,
                        float scale, int splits, float* part, hipStream_t st);
// kernels::moe_combine's arithmetic (fp32 sum of each token's pair rows in pidx order, + shared, one rounding) over
// (token, 256-dim slice) workgroups: a decode step's few tokens do not run on one workgroup each.
// hp.x set: also the mHC post-mix of the combined row into the token's streams hp.x [T][hc][dim] in place
// (hc_post_slices' arithmetic), with hp.post / hp.comb.
struct HcPostArgs {
  bf16* x = nullptr;
  const float* post = nullptr;
  const float* comb = nullptr;
  int hc = 0;
};
void glm_moe_combine(const bf16* pair_out, const int32_t* pidx, const bf16* shared, bf16* out, int T, int K, int dim,
                     hipStream_t st, const HcPostArgs& hp = {});
// Combined remote outputs (the APU's share of a step's MoE in one fp32 row per token instead of a bf16 row per pair
// over the host link). Remote side: dst[u] = sum over j of pair_out[u2p[u * K + j]] (bf16 rows, fp32, j ascending,
// -1 skipped) for u < *n_tokens. Card side: out[t] = bf16(sum of t's local pair rows (p < *remote0, pidx order) +
// partial[tok2u[t]] (if >= 0) + shared[t]), glm_moe_combine's order with the remote pairs regrouped into one fp32 term;
// partial: the remote side's pinned rows, read across the link.
void glm_remote_partial(const bf16* pair_out, const int32_t* u2p, const int* n_tokens, float* dst, int K, int dim,
                        int max_tokens, hipStream_t st);
void glm_moe_combine_partial(const bf16* pair_out, const int32_t* pidx, const int* remote0, const int32_t* tok2u,
                             const float* partial, const bf16* shared, bf16* out, int T, int K, int dim,
                             hipStream_t st, const HcPostArgs& hp = {});

// The selection as a compact token list: selected pools [n][k] (the top-k's output: valid pools first) -> idx [n][w]
// (w = k * pool + pool - 1): the pools' tokens, then the query's incomplete pool, then -1. A query at position p has
// glm_sel_count(p) entries (every complete pool is visible and scored, so min((p + 1) / pool, k) are valid).
void glm_idx_expand(const int32_t* pools, const int32_t* pos, int32_t* idx, int n, int k, int pool, hipStream_t st);
__host__ __device__ inline int glm_sel_count(int p, int k, int pool) {
  const int full = (p + 1) / pool;
  return (full < k ? full : k) * pool + (p + 1) % pool;
}

// Grouped bf16 GEMM over heads (gemm_wmma's bf16 kernel with groups): c [M, G, R] = a [M, G, K] . w[g]^T, w [G][R][K].
void gemm_grouped_bf16_wmma(const bf16* a, const bf16* w, bf16* c, int M, int G, int R, int K, hipStream_t st);

}  // namespace unholy::kernels

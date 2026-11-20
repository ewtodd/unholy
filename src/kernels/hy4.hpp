// Hy4 flash kernels (model/hy4.hpp). The arithmetic follows transformers' hy_v4_flash (PR #50903, the oracle in
// tools/ref/hy4_oracle): where it rounds to bf16 between ops these kernels round at the same point; where it computes
// in fp32 (the hyper-connections, RoPE, the indexer's pooling and scores, the router) they do too.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/exl3_moe.hpp"
#include "kernels/moe_dispatch.hpp"

namespace unholy::kernels {

using bf16 = uint16_t;

// ---- Identity hyper-connections (HYV4FlashHyperConnection: DeepSeek-V4's pre-mix without the Sinkhorn comb) --------
// Per token t, flat = x[t] as fp32 [hc * dim]: mix = (fn . flat) * rsqrt(mean(flat^2) + eps) ([2 hc]);
// pre = sigmoid(mix[:hc] * scale[0] + base[:hc]) + hc_eps; post = mag * sigmoid(mix[hc:] * scale[1] + base[hc:]) +
// hc_eps; y[t] = bf16(sum_i pre[i] * x[t][i]). fn f32 [2 hc][hc * dim], base f32 [2 hc], scale f32 [2]; post f32
// [T][hc]. With norm_w: also hn = rmsnorm_hf(y, norm_w, eps) (the layer's input / post-attention norm).
void hy4_ihc_pre(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, float* post, int T,
                 int hc, int dim, float eps, float hc_eps, float mag, hipStream_t st, const bf16* norm_w = nullptr,
                 bf16* hn = nullptr);
// x[t][i] = bf16(post[t][i] * out[t] + x[t][i]) in fp32.
void hy4_ihc_post(const bf16* out, bf16* x, const float* post, int T, int hc, int dim, hipStream_t st);
// HYV4FlashHyperHead: mix = (fn . flat) * rsqrt(mean(flat^2) + eps) ([hc]); pre = sigmoid(mix * scale + base) + eps;
// y[t] = bf16(sum_i pre[i] x[t][i]). fn f32 [hc][hc * dim], base f32 [hc], scale f32 [1].
void hy4_hc_head(const bf16* x, const float* fn, const float* scale, const float* base, bf16* y, int T, int hc,
                 int dim, float eps, float hc_eps, hipStream_t st);

// ---- RoPE ---------------------------------------------------------------------------------------------------------
// In place on bf16 rows: for row r, head h, the 64 values at x + r * ld + h * hstride + off, interleaved pairs
// (2i, 2i + 1) rotated by angle pos[r] * theta^(-2i / 64) with cos / sin rounded to bf16 (transformers' rotary
// tables are bf16), the products in fp32, one rounding. Text positions (all M-RoPE planes equal).
// pos3 (optional, [rows][3]): M-RoPE's three planes (t, h, w) instead of pos: pair i rotates by the h plane when
// i % 3 == 1 and i < 30, the w plane when i % 3 == 2 and i < 30, else t (interleaved sections [12, 10, 10]).
void hy4_rope(bf16* x, int ld, int rows, int heads, int hstride, int off, const int32_t* pos, float theta,
              hipStream_t st, const int32_t* pos3 = nullptr);
// M-RoPE positions of the pooled indexer groups: the chunk's rows [n][3] (row q - s0 for slot q), and the planes of
// the first slot of the group the chunk starts in (when that slot is before the chunk).
struct Hy4GroupRope {
  const int32_t* rows = nullptr;  // null: plain positions (slot = position)
  int g0[3] = {0, 0, 0};
};

// ---- Attention ----------------------------------------------------------------------------------------------------
// Key rows are 576 bf16: the normed latent (512) then the roped key (64). Sparse layers keep every position's row in
// the slot's pages (256 rows per page); window layers keep the last 512 positions in a per-slot ring (position % 512)
// and read the chunk's own positions from its rows.
constexpr int kHy4Row = 576, kHy4Lat = 512, kHy4PageRows = 256, kHy4Window = 512;
struct Hy4Keys {
  const bf16* pages = nullptr;     // sparse: [pages][256][576]
  const int32_t* table = nullptr;  // sparse: the slot's row of the block table
  const bf16* ring = nullptr;      // window: [512][576] of the slot
  const bf16* chunk = nullptr;     // window: the chunk's rows [n][576] (positions s0 ..)
  int s0 = 0;
  const bf16* block = nullptr;     // DFly2 draft block: rows [6][576] of positions anchor .. (before the pages)
  int anchor = 0;
};
// Absorbed MLA with learned sinks (HYV4FlashAttention, eager: softmax over the scores and the head's sink logit,
// the sink's probability dropped), H heads (32 or 64), one block per query: q bf16 [n][H][576] (W_uk-absorbed nope
// query, then the roped query), o bf16 [n][H][512] (the latent-space output, before W_uv). Keys: idx[t][0 .. w) when
// idx is given (-1 skipped), else positions [lo, pos[t]] with lo = max(0, pos[t] - window + 1) (window 0: from 0).
// sinks f32 [H]. gfx12 only.
// counts (optional, with idx): the number of valid leading entries of each query's list (default: the indexer's).
// ws (optional, ws_bytes): fp32 workspace for few queries: their keys split over several workgroups (flash-decoding),
// each split's partial output and softmax state there, then combined; hy4_attn_split_bytes(n, H) is enough for all.
void hy4_mla_attn(const bf16* q, int H, const Hy4Keys& keys, const int32_t* idx, int w, int window,
                  const int32_t* pos, const float* sinks, bf16* o, int n, float scale, hipStream_t st,
                  const int32_t* counts = nullptr, float* ws = nullptr, size_t ws_bytes = 0);
constexpr int kHy4AttnSplitRows = 64, kHy4AttnMaxSplit = 16, kHy4AttnSplitBlocks = 128;
inline size_t hy4_attn_split_bytes(int H) {
  return size_t(kHy4AttnSplitBlocks) * 2 * H * (kHy4Lat + 2) * 4;  // n * S <= 2 * kHy4AttnSplitBlocks
}
// q [T][H][256] (nope 192, roped 64) -> qn [T][H][192] (the absorbed product's input).
void hy4_split_q(const bf16* q, bf16* qn, int T, int H, hipStream_t st);
// out [T][H][576] = [qa [T][H][512], q's roped 64 (q [T][H][256], dims 192..255)].
void hy4_pack_q(const bf16* qa, const bf16* q, bf16* out, int T, int H, hipStream_t st);
// o = bf16(o * bf16(sigmoid(g))) elementwise (the attention output gate).
void hy4_gate_mul(bf16* o, const bf16* g, size_t n, hipStream_t st);
// A window layer's ring takes the chunk's last <= 512 rows (positions s0 .. s0 + n - 1), after its attention.
void hy4_ring_store(const bf16* rows, bf16* ring, int s0, int n, hipStream_t st);
// A sparse layer's rows [n][576] of positions s0 .. into the slot's pages.
void hy4_page_store(const bf16* rows, bf16* pages, const int32_t* table, int s0, int n, hipStream_t st);

// ---- Speculative decoding: per-slot state of verify chunks -------------------------------------------------------
// Window ring rows of positions s0 .. s0 + n - 1 (ring row p % 512) into bak [n][576] before a verify chunk overwrites
// them; ring_restore puts back the rows of positions [from, to) (rejected drafts).
void hy4_ring_backup(const bf16* ring, bf16* bak, int s0, int n, hipStream_t st);
void hy4_ring_restore(bf16* ring, const bf16* bak, int s0, int from, int to, hipStream_t st);
// An indexer tail after a partly rejected verify chunk at s0: the tail as it was before the chunk (bak [4][128]) with the
// accepted positions s0 .. from - 1 (their token keys vk [*][128], row q - s0) applied.
void hy4_tail_restore(bf16* tail, const bf16* bak, const bf16* vk, int s0, int from, hipStream_t st);
// DFly2's history lists: for rows [t0, t0 + n) of a chunk (positions pos[t]), the token positions the indexer recalled
// (sel [*][w], the valid ones first; null: the chunk attended densely, i.e. every position <= pos[t]) into the slot's
// ring of the last 6 positions: dst [6][2048] at row pos % 6, counts [6].
void hy4_dfly_save_sel(const int32_t* sel, int w, const int32_t* pos, int t0, int n, int32_t* dst, int32_t* counts,
                       int groups, hipStream_t st);
// The draft block's key lists: for request r (anchor a_r, history hist_r [2048] with hcount_r entries), row j: the
// history then positions a_r .. a_r + j. lists [R][6][w], counts [R][6]. hist / hcount per request (device pointers).
void hy4_dfly_lists(const int32_t* const* hist, const int32_t* const* hcount, const int32_t* anchor, int32_t* lists,
                    int32_t* counts, int R, int w, hipStream_t st);

// ---- Compressed indexer -------------------------------------------------------------------------------------------
// Pooled keys: bf16 [pages][64][128] (one per 4-token group). A slot's last <= 4 token keys sit in a tail (index
// position % 4) until their group completes.
constexpr int kHy4Pool = 4, kHy4IdxDim = 128, kHy4IdxHeads = 16;
struct Hy4Pooled {
  bf16* pk = nullptr;
  const int32_t* table = nullptr;
};
// Groups completed by positions [s0, s0 + n): the members' normed token keys (tail, then the chunk's tk [n][128]) ->
// v = Wv k, g = Wg k + ape[j] (fp32, Wv / Wg f32 [128][128], ape f32 [4][128]), softmax over the members per dim,
// pooled = sum v * p, key_norm (fp32 RMSNorm, weight f32 [128]) -> bf16, RoPE on dims 64.. at the group's first
// position (theta), stored. Then the tail takes the chunk's last <= 4 keys.
void hy4_idx_pool(const bf16* tk, bf16* tail, const float* wv, const float* wg, const float* ape, const float* knorm,
                  float eps, float theta, Hy4Pooled pg, int s0, int n, hipStream_t st, Hy4GroupRope gr = {});
// Scores of the chunk's n queries against groups [0, npool): s[t][p] = sum_h w[t][h] * relu(q[t][h] . pk[p]) with
// w = float(hw[t][h]) * wscale, -inf for p >= pos[t] / 4 (the query's own group and later). q bf16 [n][16][128]
// (roped), hw bf16 [n][16]; out f32 [n][ld]. gfx12 (WMMA).
void hy4_idx_scores(const bf16* q, const bf16* hw, float wscale, Hy4Pooled pg, const int32_t* pos, float* out, int ld,
                    int n, int npool, hipStream_t st);
// Selected groups [n][k] (top-k output, -1 padded) -> token positions idx [n][k * 4 + 4]: each valid group's 4 tokens
// (groups < pos / 4 only), then the query's own group [pos - pos % 4, pos], then -1.
void hy4_idx_expand(const int32_t* groups, const int32_t* pos, int32_t* idx, int n, int k, hipStream_t st);

// ---- Engram -------------------------------------------------------------------------------------------------------
// Rows: e4m3 codes q [T][16][96] and bf16 scales s [T][16] -> e = bf16(float(q) * s) -> per head RMSNorm (nn.RMSNorm,
// eps = fp32's epsilon) times w [16][96] -> out bf16 [T][1536]. valid [T] (0: no n-gram here, the row is unused).
void hy4_engram_rows(const uint8_t* q, const bf16* s, const bf16* w, bf16* out, int T, hipStream_t st);
// The memory's contribution added to the residual (HYV4FlashEngramLayer.forward after its projections): kv f32
// [T][5][dim] = the four key projections and the value projection of the rows (without bias), bias bf16 [5][dim];
// per stream i: key = rms(bf16(kv_i + b_i)) * nk_i, query = rms(x[t][i]) * nq_i, gate = sigmoid(sign(d) sqrt(max(|d|,
// 1e-6))) with d = sum(key * query) / sqrt(dim); main = pre_gamma * rms(bf16(kv_4 + b_4)); x[t][i] += gate * main +
// hcs_i * main (bf16 steps as transformers). Norm weights bf16 [4][dim]; rms eps fp32's epsilon. mask [T]: 0 leaves x.
void hy4_engram_mix(const float* kv, const bf16* bias, const bf16* nk, const bf16* nq, const bf16* pre_gamma,
                    const bf16* hcs, const uint8_t* mask, bf16* x, int T, int hc, int dim, hipStream_t st);

// ---- MLP / MoE ----------------------------------------------------------------------------------------------------
// h = bf16(bf16(silu(g)) * u) (the plain MLP: dense layer 0, the shared expert). g / u rows of width I (stride ld).
void hy4_swiglu(const bf16* g, const bf16* u, int ld, bf16* h, int rows, int I, hipStream_t st);
// The routed experts' gate: gu [rows][2 I] (gate, then up) -> h = bf16(bf16(silu(min(g, lim))) * clamp(u, +-lim)).
// rows: *rows_dev when given, else rows.
void hy4_swiglu_clamp(const bf16* gu, bf16* h, int rows, int I, float limit, hipStream_t st,
                      const int* rows_dev = nullptr);
// y[r] = bf16(y[r] * w[r]) (a routed pair's output times its routing weight), rows [*r0_dev, *r1_dev) or [0, rows).
void hy4_scale_rows(bf16* y, const float* w, int rows, int dim, hipStream_t st, const int* r0_dev = nullptr,
                    const int* r1_dev = nullptr);
// out[t] = bf16(sum of t's local pair rows (p < *remote0, pidx order) + remote[tok2u[t]] (fp32 rows, if >= 0)), fp32
// accumulation. remote null: every pair is local.
void hy4_moe_sum(const bf16* pair_out, const int32_t* pidx, const int* remote0, const int32_t* tok2u,
                 const float* remote, bf16* out, int T, int K, int dim, hipStream_t st);
// The remote side's pair rows from its compact token rows: xa[p] = in[p2u[p]] for p in [plan->remote0, plan->P)
// (grid sized for max_rows).
void hy4_gather_remote(const bf16* in, const int32_t* p2u, const MoePlan* plan, bf16* xa, int dim, int max_rows,
                       hipStream_t st);
// x = bf16(x + bf16(g * f)) per row (DFly2's slot-0 fusion: x embed row, g the fusion gate [dim], f eh_proj's output).
void hy4_gated_add(bf16* x, int ldx, const bf16* g, const bf16* f, int rows, int dim, hipStream_t st);
// out [rows][2 dim] = [a, b] (two normalized rows side by side).
void hy4_concat2(const bf16* a, const bf16* b, bf16* out, int rows, int dim, hipStream_t st);
// out = bf16(a + b) elementwise.
void hy4_add(const bf16* a, const bf16* b, bf16* out, size_t n, hipStream_t st);
// out[r][c] = bf16(x[r][c] + bias[c]) from an fp32 GEMM result.
void hy4_bias_bf16(const float* x, const bf16* bias, bf16* out, int rows, int cols, hipStream_t st);

// Routed experts with e4m3 weights and bf16 128x128 block scales: out[r] = xa[r] @ (bf16(w * s))^T for the rows of
// each tile (<= kHy4TileRows rows of one expert; tiles [0, *ntiles_dev) or [0, ntiles)). w_tab [E] -> e4m3 [N][K],
// s_tab [E] -> bf16 [N / 128][K / 128], null where the expert is not on this device. K, N % 128 == 0. gfx11 + gfx12.
constexpr int kHy4TileRows = 64;
void hy4_fp8_grouped_gemm(const bf16* xa, int K, const Exl3Tile* tiles, int ntiles, const uint8_t* const* w_tab,
                          const bf16* const* s_tab, bf16* out, int ldo, int N, hipStream_t st,
                          const int* ntiles_dev = nullptr);

// MXFP4 experts from the release's e4m3: rows [row0, row0 + N) of w ([*][K] e4m3, bf16 128x128 scales with kb = K / 128
// columns) as their bf16(w * s) values, quantized per 32 inputs: the e8m0 scale 2^e with e = ceil(log2(amax / 6)) or
// one below (the lower squared error), values rounded to nearest-even on the e2m1 grid (clamped to +-6). codes
// [N][K / 2] (low nibble first), scales [N][K / 32]: the checkpoint layout mx_repack expects.
void hy4_fp8_to_mxfp4(const uint8_t* w, const bf16* s, int kb, int row0, int N, int K, uint8_t* codes, uint8_t* scales,
                      hipStream_t st);
// h = bf16(bf16(silu(min(g, lim))) * clamp(u, +-lim)) from separate gate / up rows [rows][I].
void hy4_swiglu_clamp2(const bf16* g, const bf16* u, bf16* h, int rows, int I, float limit, hipStream_t st);

// The same products for tiles of <= kHy4GemvRows rows (decode): a wave per 4 output columns, K split over its lanes in
// 16-input chunks (K % 512 == 0); each weight dequantized to bf16(w * s) as above, fp32 sums. ldi: the rows' stride.
constexpr int kHy4GemvRows = 8;
void hy4_fp8_grouped_gemv(const bf16* xa, int ldi, int K, const Exl3Tile* tiles, int ntiles,
                          const uint8_t* const* w_tab, const bf16* const* s_tab, bf16* out, int ldo, int N,
                          hipStream_t st, const int* ntiles_dev = nullptr);
// The same GEMM / GEMV for NVFP4 experts (ggml's NVFP4 as AngelSlim's SPARK-4Bit GGUF has them: e2m1 values with an
// unsigned-e4m3 scale per 16 inputs, value = e2m1 * scale, exact in bf16): codes [N][K / 2] (in each group of 16, byte j
// holds values j and j + 8), scales [N][K / 16] (hy4_nvfp4_split from the GGUF blocks).
void hy4_nvfp4_grouped_gemm(const bf16* xa, int K, const Exl3Tile* tiles, int ntiles, const uint8_t* const* w_tab,
                            const uint8_t* const* s_tab, bf16* out, int ldo, int N, hipStream_t st,
                            const int* ntiles_dev = nullptr);
void hy4_nvfp4_grouped_gemv(const bf16* xa, int ldi, int K, const Exl3Tile* tiles, int ntiles,
                            const uint8_t* const* w_tab, const uint8_t* const* s_tab, bf16* out, int ldo, int N,
                            hipStream_t st, const int* ntiles_dev = nullptr);
// GGUF NVFP4 blocks of an [N][K] matrix -> codes [N][K / 2], scales [N][K / 16].
void hy4_nvfp4_split(const uint8_t* blocks, int N, int K, uint8_t* codes, uint8_t* scales, hipStream_t st);
// Tiles of one matrix (expert 0) over `rows` rows: ceil(rows / tile_rows) tiles of <= tile_rows rows, on the stream.
void hy4_dense_tiles(Exl3Tile* tiles, int rows, int tile_rows, hipStream_t st);

// fp8 e4m3 [N][K] x bf16 128x128 scales -> bf16(w * s) [N][K] (the oracle's dequantization of the dense weights).
void hy4_dequant_fp8(const uint8_t* w, const bf16* s, bf16* out, int N, int K, hipStream_t st);
// bf16 [n] -> f32 [n].
void hy4_bf16_to_f32(const bf16* x, float* y, size_t n, hipStream_t st);

}  // namespace unholy::kernels

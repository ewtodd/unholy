// Lightning Indexer on packed fp4 keys (fast path; reference: indexer_scores/topk_rows in ref.hpp).
//
// The reference rotates indexer keys and queries with the 128-point Hadamard and fake-quantizes them
// to fp4 (e2m1, one power-of-two e8m0 scale per 32 values), so storing the codes loses nothing: a key
// row is 64 bytes of nibbles + 4 scale bytes (vs 256 bytes of bf16). Scores use fp8 WMMA: an fp4 value
// times 2^-d (d = the row's largest block exponent minus its block's, <= 8) is exact in e4m3, so the
// 128-dim dot of a query and key row accumulates in one fp32 WMMA accumulator, and the two rows'
// power-of-two scales fold into the epilogue (they commute with the bf16 roundings). The ReLU, the
// head weights and the in-order head sum of the reference are kept: scores match it bit for bit.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace unholy::kernels {

using bf16 = uint16_t;

constexpr int kFp4RowsPerPage = 64;                          // indexer rows per 256-token page
constexpr int kFp4RowBytes = 64;                             // 128 nibbles
constexpr int kFp4PageBytes = kFp4RowsPerPage * (kFp4RowBytes + 4);  // codes, then e8m0 scales

// Indexer key pool: page p at pool + p * kFp4PageBytes; rows of a sequence through its block table.
struct PagedFp4 {
  uint8_t* pool = nullptr;
  const int32_t* table = nullptr;
};

// Quantizes rows [n, 128] (bf16, Hadamard-rotated) like fake_quant_fp4(block 32) and stores codes +
// scales as rows [row0, row0 + n) of the sequence.
void store_rows_fp4(const bf16* rows, int n, PagedFp4 dst, int row0, hipStream_t st);
// Rows [row0, row0 + n) back to bf16 (the values fake_quant_fp4 would have produced).
void load_rows_fp4(PagedFp4 src, int row0, int n, bf16* out, hipStream_t st);

// q [T, H, 128] bf16 (Hadamard-rotated) -> fake_quant_fp4(block 32) values as e4m3 bytes relative to the
// row's largest block scale (exact while the blocks' scales are within 2^8 of each other), q8
// [T, H, 128], and that scale's e8m0 exponent, qe [T, H].
void quant_q_fp4(const bf16* q, int T, int H, uint8_t* q8, uint8_t* qe, hipStream_t st);

// scores [T, nkeys] bf16 (bit patterns as uint16): sum_h bf16(relu(bf16(q_h . k_j)) * w_h) over heads in
// order, rounded to bf16; -inf for keys a query cannot see (j >= (pos + 1) / ratio). H must be 64.
// The query chain of the fast indexer in one launch: rope (forward, the last 2 * npair dims of each
// 128-dim head of q [T, H, 128]), hadamard128, then quant_q_fp4 into q8 / qe; q keeps the rotated bf16 values.
void index_q_chain(bf16* q, int T, int H, const int32_t* pos, const float2* freqs, int npair, uint8_t* q8, uint8_t* qe,
                   hipStream_t st);
// From kIndexerQMinRows queries (prefill) each query's codes stay in a wave's registers across 1024 keys
// instead of being re-staged per 128 keys; the same scores bit for bit (UNHOLY_INDEXER_Q=0: off).
void indexer_scores_fp4(const uint8_t* q8, const uint8_t* qe, const bf16* w, PagedFp4 keys, const int32_t* pos,
                        uint16_t* scores, int T, int H, int nkeys, int ratio, hipStream_t st);
constexpr int kIndexerQMinRows = 64;
bool indexer_q();
void indexer_force_q(int on);  // tests: 1 / 0 forces it on / off, -1 restores the default

// Top-k per row of bf16 scores, ordered (score desc, index asc); -1 padded; -inf never selected.
void topk_bf16(const uint16_t* scores, int T, int n, int k, int32_t* idx, hipStream_t st);
// Several score blocks in one launch (e.g. the decode chunks of a step): group g's rows [rows][n] at
// scores + off, its top-k rows [rows][k] at idx + idx_off; per row the same as topk_bf16.
struct TopkRows {
  size_t off;
  int rows, n, k, idx_off;
};
struct TopkBatch {
  static constexpr int kMax = 32;
  int ngroups = 0;
  TopkRows g[kMax];
};
// With scratch (topk_split_scratch_words(rows) uint32), up to kTopkSplitMaxRows rows of which the longest
// has >= kTopkSplitMinKeys keys run on several workgroups per row (decode at long context): same output.
constexpr int kTopkSplitMinKeys = 8192, kTopkSplitMaxRows = 64;
size_t topk_split_scratch_words(int rows);
void topk_bf16_batch(const uint16_t* scores, const TopkBatch& b, int32_t* idx, hipStream_t st,
                     uint32_t* scratch = nullptr, size_t scratch_words = 0);

}  // namespace unholy::kernels

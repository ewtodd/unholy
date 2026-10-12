// Lightning Indexer on packed fp4 keys (fast path; reference: indexer_scores/topk_rows in ref.hpp).
//
// The reference rotates indexer keys and queries with the 128-point Hadamard and fake-quantizes them
// to fp4 (e2m1, one power-of-two e8m0 scale per 32 values), so storing the codes loses nothing: a key
// row is 64 bytes of nibbles + 4 scale bytes (vs 256 bytes of bf16). Scores use int8 WMMA on 2*e2m1
// integers: each 32-value block's dot is an exact integer, scaled by the two exact power-of-two scales,
// so per-head dots equal the reference's fp32 sums; the bf16 roundings, the ReLU, the head weights and
// the head order of the reference are kept, so the scores are the reference's bit for bit.
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

// q [T, H, 128] bf16 (Hadamard-rotated) -> codes int8 [T, H, 128] (2 * e2m1, signed) and e8m0
// scales [T, H, 4]: the same quantization as fake_quant_fp4(q, block 32).
void quant_q_fp4(const bf16* q, int T, int H, int8_t* qc, uint8_t* qs, hipStream_t st);

// scores [T, nkeys] bf16 (bit patterns as uint16): sum_h bf16(relu(bf16(q_h . k_j)) * w_h) over heads in
// order, rounded to bf16; -inf for keys a query cannot see (j >= (pos + 1) / ratio). H must be 64.
void indexer_scores_fp4(const int8_t* qc, const uint8_t* qs, const bf16* w, PagedFp4 keys, const int32_t* pos,
                        uint16_t* scores, int T, int H, int nkeys, int ratio, hipStream_t st);

// Top-k per row of bf16 scores, ordered (score desc, index asc); -1 padded; -inf never selected.
void topk_bf16(const uint16_t* scores, int T, int n, int k, int32_t* idx, hipStream_t st);

}  // namespace unholy::kernels

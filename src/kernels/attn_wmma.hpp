// WMMA sparse attention (gfx12). Same contract as kernels::sparse_attn (ref.hpp).
#pragma once

#include "kernels/ref.hpp"

namespace unholy::kernels {

void sparse_attn_wmma(const bf16* q, const int32_t* pos, AttnKv kv, const float* sink, bf16* o, int T, int heads,
                      int dim, float scale, hipStream_t st);

// One launch for the chunks of a step (different sequences) of one layer: per query exactly what
// sparse_attn_wmma computes for the chunk's own call. q / pos / o / vis are the batch's ([T] rows);
// chunk c's new kv rows are kv rows [t0, t0 + n), its window ring ring + slot * ring_stride, its
// compressed rows through block-table row tables + slot * max_pages, its top-k rows
// comp_idx + idx_off ([n][ncomp]; ncomp 0: every visible row).
struct AttnChunk {
  int t0, n, s0, slot, ncomp, idx_off;
};
struct AttnBatch {
  static constexpr int kMax = 32;
  const bf16* kv = nullptr;
  const bf16* ring = nullptr;
  size_t ring_stride = 0;
  bf16* comp_pool = nullptr;
  const int32_t* tables = nullptr;
  int max_pages = 0, rows_per_page = 0, row_stride = 0, ratio = 0;
  const int32_t* comp_idx = nullptr;
  const int32_t* vis = nullptr;
  int nchunks = 0;
  AttnChunk c[kMax];
};
void sparse_attn_wmma_batch(const bf16* q, const int32_t* pos, const AttnBatch& b, const float* sink, bf16* o,
                            float scale, hipStream_t st);

}  // namespace unholy::kernels

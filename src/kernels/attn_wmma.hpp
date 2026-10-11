// WMMA sparse attention (gfx12). Same contract as kernels::sparse_attn (ref.hpp).
#pragma once

#include "kernels/ref.hpp"

namespace unholy::kernels {

void sparse_attn_wmma(const bf16* q, const int32_t* pos, AttnKv kv, const float* sink, bf16* o, int T, int heads,
                      int dim, float scale, hipStream_t st);

}  // namespace unholy::kernels

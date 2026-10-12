// Device-side MoE dispatch (see moe_dispatch.hip). All sizes that depend on routing stay on the device;
// consumers launch for the maximum and read the real extents from MoePlan.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/exl3_moe.hpp"

namespace unholy::kernels {

struct MoePlan {
  int P;               // pair rows (T * K)
  int remote0;         // first row of the remote block
  int n_local_tiles;   // tiles in local_tiles
  int n_remote_tiles;  // tiles in remote_tiles
};

// count[e] = pairs routed to e; also accumulated into hist[e] (a running routing histogram) if given.
void moe_count(const int32_t* sel, int T, int K, int32_t* count, int E, hipStream_t st, uint32_t* hist = nullptr);
// `resident` [E]: 1 = this device holds the expert. Also clears `fill`.
void moe_plan(const int32_t* count, const uint8_t* resident, int E, int P, MoePlan* plan, Exl3Tile* local_tiles,
              Exl3Tile* remote_tiles, int32_t* off, int32_t* fill, int max_tiles, hipStream_t st);
void moe_scatter(const int32_t* sel, const float* wt, int T, int K, const int32_t* off, int32_t* fill, int32_t* eidx,
                 float* ewt, int32_t* row_expert, int32_t* pidx, hipStream_t st);
// Rows [remote0, P) (and their weights / experts) between two buffers with the same row indexing,
// e.g. device pair rows -> pinned host staging; grids sized for max_rows.
void copy_remote_rows(const uint16_t* src, uint16_t* dst, const MoePlan* plan, int dim, int max_rows, hipStream_t st);
void copy_remote_meta(const float* ewt, const int32_t* row_expert, float* w_dst, int32_t* e_dst, const MoePlan* plan,
                      int max_rows, hipStream_t st);

}  // namespace unholy::kernels

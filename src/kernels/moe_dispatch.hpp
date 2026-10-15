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
  int n_remote_tokens; // tokens with >= 1 remote pair (remote_tokens; 0 otherwise)
};

// count[e] = pairs routed to e; also accumulated into hist[e] (a running routing histogram) if given.
void moe_count(const int32_t* sel, int T, int K, int32_t* count, int E, hipStream_t st, uint32_t* hist = nullptr);
// `resident` [E]: 1 = this device holds the expert. Also clears `fill`. Tiles hold <= tile_rows rows.
void moe_plan(const int32_t* count, const uint8_t* resident, int E, int P, MoePlan* plan, Exl3Tile* local_tiles,
              Exl3Tile* remote_tiles, int32_t* off, int32_t* fill, int max_tiles, hipStream_t st,
              int tile_rows = kExl3TileRows);
void moe_scatter(const int32_t* sel, const float* wt, int T, int K, const int32_t* off, int32_t* fill, int32_t* eidx,
                 float* ewt, int32_t* row_expert, int32_t* pidx, hipStream_t st);
// Rows [remote0, P) (and their weights / experts) between two buffers with the same row indexing,
// e.g. device pair rows -> pinned host staging; grids sized for max_rows.
// The remote side's per-layer inputs (plan, remote tiles, rows' routing weights and experts) into its
// staging, by a kernel: stream-ordered with the flag packets around it.
void stage_remote(const MoePlan* plan, const Exl3Tile* rtiles, const float* ewt, const int32_t* row_expert,
                  MoePlan* plan_dst, Exl3Tile* tiles_dst, float* w_dst, int32_t* e_dst, int max_rows, hipStream_t st);
void copy_remote_rows(const uint16_t* src, uint16_t* dst, const MoePlan* plan, int dim, int max_rows, hipStream_t st);
// Compact remote inputs (native MXFP4 experts): the tokens with at least one remote pair, in token
// order. tok2u [T] (-1: none), u2t [T] (compact -> token), plan->n_remote_tokens, and p2u_dst[p] = the
// compact index of pair row p's token for the remote rows [remote0, P). One block.
// With u2p_dst (combined outputs): u2p_dst[u * K + j] = token u's j-th pair row if remote, else -1.
void remote_tokens(const int32_t* pidx, const int32_t* eidx, int T, int K, MoePlan* plan, int32_t* tok2u,
                   int32_t* u2t, int32_t* p2u_dst, int32_t* u2p_dst, hipStream_t st);
// Remote side, combined outputs: dst[u] = bf16(sum over token u's remote pair rows of pair_out, in fp32,
// in ascending expert order) for u < plan->n_remote_tokens. One bf16 row per token crosses the host
// link instead of one per remote pair; the remote share is rounded to bf16 once (the reference model
// accumulates every expert's output in bf16).
void remote_combine(const uint16_t* pair_out, const int32_t* u2p, const MoePlan* plan, uint16_t* dst, int K, int dim,
                    int max_tokens, hipStream_t st);
// Card side: those rows (dst of remote_combine, in host memory) -> rows [remote0, remote0 + U) of dst.
void copy_remote_partial(const uint16_t* src, uint16_t* dst, const MoePlan* plan, int dim, int max_tokens,
                         hipStream_t st);
// out[t] = bf16(sum of token t's local pair rows (pidx order) + its remote partial row (pair_out row
// remote0 + tok2u[t]) + shared[t]), fp32 accumulation.
void moe_combine_partial(const uint16_t* pair_out, const int32_t* pidx, const MoePlan* plan, const int32_t* tok2u,
                         const uint16_t* shared, uint16_t* out, int T, int K, int dim, hipStream_t st);
// Those tokens' rows of x [T, dim] as the experts' input transform fake-quantizes them (mx_xform_in:
// e4m3 per value, e8m0 scale per 128) -> codes [U, dim] + scale exponents [U, dim / 128]. The remote side
// rebuilds exactly the bf16 values it would have computed from the pair rows, from 1 + 1/128 bytes per
// value of one row per token instead of 2 bytes per value of one row per remote pair.
void quant_remote_tokens(const uint16_t* x, int dim, const int32_t* u2t, const MoePlan* plan, uint8_t* q_dst,
                         uint8_t* s_dst, int max_tokens, hipStream_t st);
void copy_remote_meta(const float* ewt, const int32_t* row_expert, float* w_dst, int32_t* e_dst, const MoePlan* plan,
                      int max_rows, hipStream_t st);

}  // namespace unholy::kernels

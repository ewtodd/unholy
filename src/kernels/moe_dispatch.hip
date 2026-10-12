// Device-side MoE dispatch: everything the host used to do between the router and the expert GEMMs.
//
//   moe_count     sel [T, K] -> count [E]
//   moe_plan      count + resident mask -> pair offsets (resident experts first, then remote, each in
//                 ascending expert order), remote0, and the <= kExl3TileRows-row tile lists of both groups
//   moe_scatter   per token: its K picks sorted by expert id; each pick gets a pair row
//                 (off[e] + atomic fill): eidx (token), ewt (weight), row_expert, pidx[t][j]
//
// The row order inside an expert's block depends on atomic arrival order; every expert kernel is
// row-independent and the combine follows pidx (ascending expert per token), so results do not.
#include "kernels/dev.hip.hpp"
#include "kernels/moe_dispatch.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

namespace {

__global__ void moe_count_kernel(const int32_t* sel, int n, int32_t* count, uint32_t* hist) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&count[sel[i]], 1);
  if (hist) atomicAdd(&hist[sel[i]], 1u);
}

// One block of 256 threads (E <= 256): serial scans by thread 0 are trivial at E = 256.
__global__ void moe_plan_kernel(const int32_t* count, const uint8_t* resident, int E, int P, MoePlan* plan,
                                Exl3Tile* local_tiles, Exl3Tile* remote_tiles, int32_t* off, int32_t* fill,
                                int max_tiles) {
  if (threadIdx.x == 0) {
    int next = 0, nl = 0, nr = 0, remote0 = 0;
    for (int pass = 0; pass < 2; ++pass) {
      if (pass == 1) remote0 = next;
      for (int e = 0; e < E; ++e) {
        if ((resident[e] != 0) != (pass == 0)) continue;
        const int c = count[e];
        off[e] = next;
        for (int i = 0; i < c; i += kExl3TileRows) {
          const Exl3Tile t{e, next + i, min(kExl3TileRows, c - i)};
          if (pass == 0 && nl < max_tiles) local_tiles[nl++] = t;
          if (pass == 1 && nr < max_tiles) remote_tiles[nr++] = t;
        }
        next += c;
      }
    }
    plan->P = P;
    plan->remote0 = remote0;
    plan->n_local_tiles = nl;
    plan->n_remote_tiles = nr;
  }
  for (int e = threadIdx.x; e < E; e += blockDim.x) fill[e] = 0;
}

__global__ void moe_scatter_kernel(const int32_t* sel, const float* wt, int T, int K, const int32_t* off, int32_t* fill,
                                   int32_t* eidx, float* ewt, int32_t* row_expert, int32_t* pidx) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int es[8];
  float ws[8];
  for (int j = 0; j < K; ++j) {
    es[j] = sel[size_t(t) * K + j];
    ws[j] = wt[size_t(t) * K + j];
  }
  for (int a = 1; a < K; ++a)  // insertion sort by expert id
    for (int b = a; b > 0 && es[b - 1] > es[b]; --b) {
      const int te = es[b];
      es[b] = es[b - 1];
      es[b - 1] = te;
      const float tw = ws[b];
      ws[b] = ws[b - 1];
      ws[b - 1] = tw;
    }
  for (int j = 0; j < K; ++j) {
    const int e = es[j];
    const int p = off[e] + atomicAdd(&fill[e], 1);
    eidx[p] = t;
    ewt[p] = ws[j];
    row_expert[p] = e;
    pidx[size_t(t) * K + j] = p;
  }
}

// Rows [plan->remote0, plan->P) of src -> dst (same row indices). Grid sized for the maximum.
__global__ void copy_remote_rows_kernel(const uint16_t* src, uint16_t* dst, const MoePlan* plan, int dim) {
  const int r0 = plan->remote0, r1 = plan->P;
  const size_t n = size_t(r1 - r0) * dim / 8;  // uint4 units
  const uint4* s = reinterpret_cast<const uint4*>(src + size_t(r0) * dim);
  uint4* d = reinterpret_cast<uint4*>(dst + size_t(r0) * dim);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    d[i] = s[i];
  __threadfence_system();
}

__global__ void copy_remote_meta_kernel(const float* ewt, const int32_t* row_expert, float* w_dst, int32_t* e_dst,
                                        const MoePlan* plan) {
  const int r0 = plan->remote0, r1 = plan->P;
  for (int i = r0 + blockIdx.x * blockDim.x + threadIdx.x; i < r1; i += gridDim.x * blockDim.x) {
    w_dst[i] = ewt[i];
    e_dst[i] = row_expert[i];
  }
  __threadfence_system();
}

}  // namespace

void moe_count(const int32_t* sel, int T, int K, int32_t* count, int E, hipStream_t st, uint32_t* hist) {
  UH_CHECK(hipMemsetAsync(count, 0, size_t(E) * 4, st));
  const int n = T * K;
  moe_count_kernel<<<(n + 255) / 256, 256, 0, st>>>(sel, n, count, hist);
  UH_CHECK_LAUNCH();
}

void moe_plan(const int32_t* count, const uint8_t* resident, int E, int P, MoePlan* plan, Exl3Tile* local_tiles,
              Exl3Tile* remote_tiles, int32_t* off, int32_t* fill, int max_tiles, hipStream_t st) {
  moe_plan_kernel<<<1, 256, 0, st>>>(count, resident, E, P, plan, local_tiles, remote_tiles, off, fill, max_tiles);
  UH_CHECK_LAUNCH();
}

void moe_scatter(const int32_t* sel, const float* wt, int T, int K, const int32_t* off, int32_t* fill, int32_t* eidx,
                 float* ewt, int32_t* row_expert, int32_t* pidx, hipStream_t st) {
  moe_scatter_kernel<<<(T + 63) / 64, 64, 0, st>>>(sel, wt, T, K, off, fill, eidx, ewt, row_expert, pidx);
  UH_CHECK_LAUNCH();
}

void copy_remote_rows(const uint16_t* src, uint16_t* dst, const MoePlan* plan, int dim, int max_rows, hipStream_t st) {
  const size_t units = size_t(max_rows) * dim / 8;
  const int blocks = int(std::min<size_t>((units + 255) / 256, 1024));
  copy_remote_rows_kernel<<<std::max(blocks, 1), 256, 0, st>>>(src, dst, plan, dim);
  UH_CHECK_LAUNCH();
}

void copy_remote_meta(const float* ewt, const int32_t* row_expert, float* w_dst, int32_t* e_dst, const MoePlan* plan,
                      int max_rows, hipStream_t st) {
  copy_remote_meta_kernel<<<std::max(1, std::min((max_rows + 255) / 256, 256)), 256, 0, st>>>(ewt, row_expert, w_dst,
                                                                                               e_dst, plan);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

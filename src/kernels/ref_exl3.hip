// Reference EXL3 linear: y = H(H(x * suh) W_inner) * svh in fp32, with W_inner decoded from the
// trellis inside the GEMM's weight loader (format: src/model/exl3.hpp). The fast kernels decode tiles
// into LDS and use WMMA; this one is the arithmetic they are checked against.
#include <hip/hip_fp16.h>

#include "kernels/dev.hip.hpp"
#include "kernels/ref.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

inline int cdiv(size_t a, int b) { return int((a + b - 1) / b); }

// Inverse of the tensor-core tile permutation: row-major tile position -> trellis step.
__constant__ uint8_t kPermInv[256];

bool g_perm_ready[64] = {};

void ensure_perm(hipStream_t) {
  int dev = 0;
  UH_CHECK(hipGetDevice(&dev));
  if (g_perm_ready[dev]) return;
  uint8_t inv[256];
  for (int t = 0; t < 32; ++t) {
    const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
    const int c0 = t / 4, c1 = c0 + 8;
    const int pos[8] = {r0 * 16 + c0, r1 * 16 + c0, r2 * 16 + c0, r3 * 16 + c0,
                        r0 * 16 + c1, r1 * 16 + c1, r2 * 16 + c1, r3 * 16 + c1};
    for (int j = 0; j < 8; ++j) inv[pos[j]] = uint8_t(t * 8 + j);
  }
  UH_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(kPermInv), inv, sizeof inv));
  g_perm_ready[dev] = true;
}

__device__ __forceinline__ float decode_mul1(uint32_t state) {
  const uint32_t x = state * 0x83DCD12Du;
  const uint32_t sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24) + 0x6400u;
  const __half h = __ushort_as_half(uint16_t(sum));
  return __half2float(__hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931)));
}

// W_inner[k][n] for the GEMM loader convention W(n, k).
struct LoadExl3 {
  const uint16_t* trellis;
  int bits, N;
  __device__ float operator()(int, int n, int k) const {
    const int tiles_n = N / 16;
    const uint16_t* tile = trellis + (size_t(k / 16) * tiles_n + n / 16) * 16 * bits;
    const int t = kPermInv[(k % 16) * 16 + (n % 16)];
    const int len = 256 * bits;
    const int end = (t + 1) * bits;
    uint32_t s = 0;
    for (int i = 0; i < 16; ++i) {
      const int p = (end - 16 + i + len) % len;
      // little-endian uint32 word p/32 = (u16[2w+1] << 16) | u16[2w]; bit 31 - p%32 of it.
      const int w = p / 32, b = 31 - p % 32;
      const uint32_t word = (uint32_t(tile[2 * w + 1]) << 16) | tile[2 * w];
      s = (s << 1) | ((word >> b) & 1u);
    }
    return decode_mul1(s);
  }
};

struct LoadF32Rows {
  const float* p;
  int ld;
  __device__ float operator()(int, int r, int k) const { return p[size_t(r) * ld + k]; }
};
struct One {
  __device__ float operator()(int, int, int, int) const { return 1.0f; }
};
struct StoreF32Rows {
  float* c;
  int ldc;
  __device__ void operator()(int, int m, int n, float v) const { c[size_t(m) * ldc + n] = v; }
};

constexpr int BM = 64, BN = 64, BK = 32, NT = 256;

template <class LA, class LW, class STO>
__global__ __launch_bounds__(NT) void gemm_kernel(int M, int N, int K, LA la, LW lw, STO sto) {
  __shared__ float As[BK][BM + 1];
  __shared__ float Ws[BK][BN + 1];
  const int tm = threadIdx.x / 16, tn = threadIdx.x % 16;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = threadIdx.x; i < BM * BK; i += NT) {
      const int mm = i / BK, kk = i % BK;
      As[kk][mm] = (m0 + mm < M) ? la(0, m0 + mm, k0 + kk) : 0.0f;
    }
    for (int i = threadIdx.x; i < BN * BK; i += NT) {
      const int nn = i / BK, kk = i % BK;
      Ws[kk][nn] = (n0 + nn < N) ? lw(0, n0 + nn, k0 + kk) : 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < BK; ++kk)
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(As[kk][tm * 4 + i], Ws[kk][tn * 4 + j], acc[i][j]);
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      const int m = m0 + tm * 4 + i, n = n0 + tn * 4 + j;
      if (m < M && n < N) sto(0, m, n, acc[i][j]);
    }
}

// Row-wise: v = (in * pre) (pre optional), FWHT-128 per block, * 128^-0.5, * post (optional).
// One block of 128 threads per (row, 128-segment).
template <typename TIn, typename TOut>
__global__ void had_rows_kernel(const TIn* in, int ldi, const uint16_t* pre, const uint16_t* post, TOut* out, int ldo,
                                int cols) {
  __shared__ float b[128];
  const int row = blockIdx.y, seg = blockIdx.x, i = threadIdx.x;
  float v;
  if constexpr (sizeof(TIn) == 2) v = bf16_to_f32(in[size_t(row) * ldi + seg * 128 + i]);
  else v = in[size_t(row) * ldi + seg * 128 + i];
  if (pre) v *= __half2float(__ushort_as_half(pre[seg * 128 + i]));
  b[i] = v;
  __syncthreads();
  for (int h = 1; h < 128; h <<= 1) {
    float nv = 0.0f;
    if ((i & h) == 0) nv = b[i] + b[i + h];
    else nv = b[i - h] - b[i];
    __syncthreads();
    b[i] = nv;
    __syncthreads();
  }
  v = b[i] * 0.08838834764831845f;
  if (post) v *= __half2float(__ushort_as_half(post[seg * 128 + i]));
  if constexpr (sizeof(TOut) == 2) out[size_t(row) * ldo + seg * 128 + i] = f32_to_bf16(v);
  else out[size_t(row) * ldo + seg * 128 + i] = v;
}

}  // namespace

void exl3_in(const bf16* x, int ldx, const uint16_t* suh, float* xh, int M, int K, hipStream_t st) {
  if (M <= 0) return;
  had_rows_kernel<bf16, float><<<dim3(K / 128, M), 128, 0, st>>>(x, ldx, suh, nullptr, xh, K, K);
  UH_CHECK_LAUNCH();
}

void gemm_exl3_inner(const float* a, const uint16_t* trellis, int bits, float* c, int M, int N, int K, hipStream_t st) {
  if (M <= 0) return;
  ensure_perm(st);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_kernel<<<grid, NT, 0, st>>>(M, N, K, LoadF32Rows{a, K}, LoadExl3{trellis, bits, N}, StoreF32Rows{c, N});
  UH_CHECK_LAUNCH();
}

void exl3_out(const float* y, const uint16_t* svh, bf16* out, int ldo, int M, int N, hipStream_t st) {
  if (M <= 0) return;
  had_rows_kernel<float, bf16><<<dim3(N / 128, M), 128, 0, st>>>(y, N, nullptr, svh, out, ldo, N);
  UH_CHECK_LAUNCH();
}

}  // namespace unholy::kernels

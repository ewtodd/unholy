// Reference GEMMs: LDS-tiled fp32 FMA, with the reference's K-block partial-sum structure.
#include "kernels/dev.hip.hpp"
#include "kernels/ref.hpp"
#include "rt/hip.hpp"

namespace unholy::kernels {

using namespace unholy::dev;

namespace {

constexpr int BM = 64, BN = 64, BK = 32, NT = 256;

// C = sum over K-blocks of size G of (sum_{k in block} A(m,k) W(n,k)) * S(m, n, block).
template <class LA, class LW, class SC, class STO>
__global__ __launch_bounds__(NT) void gemm_ref_kernel(int M, int N, int K, int G, LA la, LW lw, SC sc, STO sto) {
  __shared__ float As[BK][BM + 1];
  __shared__ float Ws[BK][BN + 1];
  const int tm = threadIdx.x / 16, tn = threadIdx.x % 16;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int g = blockIdx.z;  // group (grouped GEMM), 0 otherwise
  float acc[4][4] = {};
  for (int kb = 0; kb < K; kb += G) {
    float part[4][4] = {};
    for (int k0 = kb; k0 < kb + G; k0 += BK) {
      for (int i = threadIdx.x; i < BM * BK; i += NT) {
        const int mm = i / BK, kk = i % BK;
        As[kk][mm] = (m0 + mm < M) ? la(g, m0 + mm, k0 + kk) : 0.0f;
      }
      for (int i = threadIdx.x; i < BN * BK; i += NT) {
        const int nn = i / BK, kk = i % BK;
        Ws[kk][nn] = (n0 + nn < N) ? lw(g, n0 + nn, k0 + kk) : 0.0f;
      }
      __syncthreads();
#pragma unroll 4
      for (int kk = 0; kk < BK; ++kk) {
        float a[4], w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = As[kk][tm * 4 + i];
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j] = Ws[kk][tn * 4 + j];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j) part[i][j] = fmaf(a[i], w[j], part[i][j]);
      }
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int m = m0 + tm * 4 + i, n = n0 + tn * 4 + j;
        if (m < M && n < N) acc[i][j] += part[i][j] * sc(g, m, n, kb / G);
      }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int m = m0 + tm * 4 + i, n = n0 + tn * 4 + j;
      if (m < M && n < N) sto(g, m, n, acc[i][j]);
    }
}

template <class LA, class LW, class SC, class STO>
void launch(int M, int N, int K, int G, int groups, LA la, LW lw, SC sc, STO sto, hipStream_t st) {
  if (K % BK != 0 || G % BK != 0 || K % G != 0) throw std::runtime_error("gemm_ref: K/G not multiples of 32");
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, groups);
  gemm_ref_kernel<<<grid, NT, 0, st>>>(M, N, K, G, la, lw, sc, sto);
  UH_CHECK_LAUNCH();
}

struct LoadFp8 {
  const uint8_t* p;
  int ld;
  __device__ float operator()(int, int r, int k) const { return e4m3_to_f32(p[size_t(r) * ld + k]); }
};
struct LoadE2m1 {
  const uint8_t* p;
  int K;
  __device__ float operator()(int, int r, int k) const {
    const uint8_t b = p[size_t(r) * (K / 2) + k / 2];
    return e2m1_to_f32((k & 1) ? (b >> 4) : (b & 0xf));
  }
};
// The tile-major MXFP4 layout of kernels/mx_moe.hpp.
struct LoadE2m1Tiled {
  const uint8_t* p;
  int N;
  __device__ float operator()(int, int n, int k) const {
    const size_t chunk = size_t(k / 64) * (N / 16) + n / 16;
    const uint8_t b = p[chunk * 512 + (n % 16) * 32 + (k % 64) / 2];
    return e2m1_to_f32((k & 1) ? (b >> 4) : (b & 0xf));
  }
};
struct LoadBf16 {
  const bf16* p;
  int ld;
  __device__ float operator()(int, int r, int k) const { return bf16_to_f32(p[size_t(r) * ld + k]); }
};
struct LoadF32 {
  const float* p;
  int ld;
  __device__ float operator()(int, int r, int k) const { return p[size_t(r) * ld + k]; }
};
// Grouped A: a[m][g][k].
struct LoadBf16Grouped {
  const bf16* p;
  int G, K;
  __device__ float operator()(int g, int m, int k) const { return bf16_to_f32(p[(size_t(m) * G + g) * K + k]); }
};
// Grouped W: rows g*R + r of an fp8 [G*R, K] matrix with 128x128 scales, dequantized (exact in bf16).
struct LoadFp8DequantGrouped {
  const uint8_t* w;
  const uint8_t* sw;
  int R, K;
  __device__ float operator()(int g, int r, int k) const {
    const int row = g * R + r;
    return e4m3_to_f32(w[size_t(row) * K + k]) * e8m0_to_f32(sw[size_t(row / 128) * (K / 128) + k / 128]);
  }
};

struct ScaleOne {
  __device__ float operator()(int, int, int, int) const { return 1.0f; }
};
struct ScaleW8A8 {
  const uint8_t* sa;
  const uint8_t* sw;
  int kblocks;
  __device__ float operator()(int, int m, int n, int kb) const {
    return e8m0_to_f32(sa[size_t(m) * kblocks + kb]) * e8m0_to_f32(sw[size_t(n / 128) * kblocks + kb]);
  }
};
struct ScaleA8W4 {
  const uint8_t* sa;
  const uint8_t* sw;
  int K;
  __device__ float operator()(int, int m, int n, int kb32) const {
    return e8m0_to_f32(sa[size_t(m) * (K / 128) + kb32 / 4]) * e8m0_to_f32(sw[size_t(n) * (K / 32) + kb32]);
  }
};

struct ScaleA8W4Tiled {
  const uint8_t* sa;
  const uint8_t* sw;
  int K, N;
  __device__ float operator()(int, int m, int n, int kb32) const {
    const size_t chunk = size_t(kb32 / 2) * (N / 16) + n / 16;
    return e8m0_to_f32(sa[size_t(m) * (K / 128) + kb32 / 4]) * e8m0_to_f32(sw[chunk * 32 + (n % 16) * 2 + kb32 % 2]);
  }
};

struct StoreBf16 {
  bf16* c;
  int ldc;
  __device__ void operator()(int, int m, int n, float v) const { c[size_t(m) * ldc + n] = f32_to_bf16(v); }
};
struct StoreF32 {
  float* c;
  int ldc;
  __device__ void operator()(int, int m, int n, float v) const { c[size_t(m) * ldc + n] = v; }
};
struct StoreBf16Grouped {
  bf16* c;
  int G, R;
  __device__ void operator()(int g, int m, int r, float v) const { c[(size_t(m) * G + g) * R + r] = f32_to_bf16(v); }
};

}  // namespace

void gemm_w8a8(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc, int M,
               int N, int K, hipStream_t st) {
  launch(M, N, K, 128, 1, LoadFp8{a, K}, LoadFp8{w, K}, ScaleW8A8{sa, sw, K / 128}, StoreBf16{c, ldc}, st);
}

void gemm_a8w4(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc, int M,
               int N, int K, hipStream_t st) {
  launch(M, N, K, 32, 1, LoadFp8{a, K}, LoadE2m1{w, K}, ScaleA8W4{sa, sw, K}, StoreBf16{c, ldc}, st);
}

void gemm_a8w4_tiled(const uint8_t* a, const uint8_t* sa, const uint8_t* w, const uint8_t* sw, bf16* c, int ldc,
                     int M, int N, int K, hipStream_t st) {
  launch(M, N, K, 32, 1, LoadFp8{a, K}, LoadE2m1Tiled{w, N}, ScaleA8W4Tiled{sa, sw, K, N}, StoreBf16{c, ldc}, st);
}

void gemm_bf16(const bf16* a, int lda, const bf16* w, bf16* c, int ldc, int M, int N, int K, hipStream_t st) {
  launch(M, N, K, K, 1, LoadBf16{a, lda}, LoadBf16{w, K}, ScaleOne{}, StoreBf16{c, ldc}, st);
}

void gemm_f32(const float* a, int lda, const bf16* w, float* c, int ldc, int M, int N, int K, hipStream_t st) {
  launch(M, N, K, K, 1, LoadF32{a, lda}, LoadBf16{w, K}, ScaleOne{}, StoreF32{c, ldc}, st);
}

void gemm_grouped_fp8w(const bf16* a, const uint8_t* w, const uint8_t* sw, bf16* c, int M, int G, int R, int K,
                       hipStream_t st) {
  launch(M, R, K, K, G, LoadBf16Grouped{a, G, K}, LoadFp8DequantGrouped{w, sw, R, K}, ScaleOne{},
         StoreBf16Grouped{c, G, R}, st);
}

}  // namespace unholy::kernels

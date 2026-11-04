// Benchmark of gufo's FP16 routed-expert kernel (ds41_v41_routed_f16, GGUF IQ weights) on one GPU, for comparing with
// unholy-mx-test's MXFP4 routed GEMM on the same slot counts. Not in CMake: tools/box/gufo-routed.sh compiles it
// together with gufo's ds41_routed_f16.hip.cpp (MIT, from a checkout, not vendored).
//   gufo-routed-bench [--counts FILE | --experts E --tokens T] [--type 16] [--down-type 22] [--K 5120] [--M 2304]
//                     [--reps 5] [--dev 0]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "ds41_v41.h"

#define HIP_OK(x)                                                                                    \
  do {                                                                                               \
    hipError_t e_ = (x);                                                                             \
    if (e_ != hipSuccess) {                                                                          \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));        \
      std::exit(1);                                                                                  \
    }                                                                                                \
  } while (0)

namespace {

// Bytes per 256-value block of a GGML type, and where its FP16 scales sit (-1: none).
struct TypeInfo {
  int bytes, d_off, dmin_off;
};
bool type_info(int t, TypeInfo& ti) {
  switch (t) {
    case 16: ti = {66, 0, -1}; return true;   // IQ2_XXS
    case 17: ti = {74, 0, -1}; return true;   // IQ2_XS
    case 22: ti = {82, 0, -1}; return true;   // IQ2_S
    case 19: ti = {50, 0, -1}; return true;   // IQ1_S
    case 29: ti = {56, -1, -1}; return true;  // IQ1_M
    case 18: ti = {98, 0, -1}; return true;   // IQ3_XXS
    case 10: ti = {84, 80, 82}; return true;  // Q2_K
    default: return false;
  }
}

// One expert's weights [rows][cols] as random bytes with sane scales.
void fill_expert(std::vector<uint8_t>& buf, const TypeInfo& ti, std::mt19937_64& rng) {
  for (size_t i = 0; i + 8 <= buf.size(); i += 8) {
    uint64_t v = rng();
    std::memcpy(&buf[i], &v, 8);
  }
  for (size_t i = buf.size() & ~size_t(7); i < buf.size(); i++) buf[i] = uint8_t(rng());
  for (size_t b = 0; b + ti.bytes <= buf.size(); b += ti.bytes) {
    if (ti.d_off >= 0) { buf[b + ti.d_off] = 0x00; buf[b + ti.d_off + 1] = 0x1C; }
    if (ti.dmin_off >= 0) { buf[b + ti.dmin_off] = 0x00; buf[b + ti.dmin_off + 1] = 0x1C; }
  }
}

uint16_t half_bits(float f) {
  _Float16 h = static_cast<_Float16>(f);
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

template <class F>
float time_ms(F&& f, int reps) {
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipEventRecord(a));
  for (int i = 0; i < reps; i++) f();
  HIP_OK(hipEventRecord(b));
  HIP_OK(hipEventSynchronize(b));
  float ms = 0;
  HIP_OK(hipEventElapsedTime(&ms, a, b));
  HIP_OK(hipEventDestroy(a));
  HIP_OK(hipEventDestroy(b));
  return ms / float(reps);
}

}  // namespace

int main(int argc, char** argv) {
  std::string counts_file;
  int experts = 384, tokens = 2048, type = 16, down_type = 22, reps = 5, dev = 0;
  uint32_t K = 5120, M = 2304;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto val = [&]() -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (a == "--counts") counts_file = val();
    else if (a == "--experts") experts = std::atoi(val());
    else if (a == "--tokens") tokens = std::atoi(val());
    else if (a == "--type") type = std::atoi(val());
    else if (a == "--down-type") down_type = std::atoi(val());
    else if (a == "--K") K = uint32_t(std::atoi(val()));
    else if (a == "--M") M = uint32_t(std::atoi(val()));
    else if (a == "--reps") reps = std::atoi(val());
    else if (a == "--dev") dev = std::atoi(val());
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  TypeInfo gi, di;
  if (!type_info(type, gi) || !type_info(down_type, di)) { std::fprintf(stderr, "unsupported type\n"); return 2; }
  if (K % 256 || M % 256 || reps < 1) { std::fprintf(stderr, "K and M must be multiples of 256, reps >= 1\n"); return 2; }

  std::vector<int> counts;
  std::mt19937_64 rng(12345);
  if (!counts_file.empty()) {
    std::ifstream f(counts_file);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", counts_file.c_str()); return 1; }
    int c;
    while (f >> c) counts.push_back(c);
    if (counts.empty()) { std::fprintf(stderr, "no counts in %s\n", counts_file.c_str()); return 1; }
  } else {
    counts.assign(size_t(experts), 0);
    std::uniform_int_distribution<int> pick(0, experts - 1);
    for (int s = 0; s < tokens * 6; s++) counts[size_t(pick(rng))]++;
  }
  const uint32_t E = uint32_t(counts.size());
  std::vector<int32_t> ids;
  for (uint32_t e = 0; e < E; e++) ids.insert(ids.end(), size_t(counts[e]), int32_t(e));
  const uint32_t S = uint32_t(ids.size());
  if (S == 0) { std::fprintf(stderr, "no slots\n"); return 1; }

  HIP_OK(hipSetDevice(dev));
  std::vector<int32_t> h_rows(S), h_bounds(E + 1), h_tiles(S / 64 + E + 1);
  const uint32_t n_tiles = ds41_v41_routed_plan(ids.data(), S, E, h_rows.data(), h_bounds.data(), h_tiles.data());

  const uint64_t gate_bytes = uint64_t(M) * K / 256 * gi.bytes;  // per expert, gate and up each
  const uint64_t down_bytes = uint64_t(K) * M / 256 * di.bytes;
  void *w_gate, *w_up, *w_down;
  uint16_t *x, *y16;
  float *y32, *route_w;
  int32_t *rows, *bounds, *tiles;
  HIP_OK(hipMalloc(&w_gate, gate_bytes * E));
  HIP_OK(hipMalloc(&w_up, gate_bytes * E));
  HIP_OK(hipMalloc(&w_down, down_bytes * E));
  HIP_OK(hipMalloc(&x, size_t(S) * K * 2));
  HIP_OK(hipMalloc(&y16, size_t(S) * M * 2));
  HIP_OK(hipMalloc(&y32, size_t(S) * K * 4));
  HIP_OK(hipMalloc(&route_w, size_t(S) * 4));
  HIP_OK(hipMalloc(&rows, S * 4));
  HIP_OK(hipMalloc(&bounds, (E + 1) * 4));
  HIP_OK(hipMalloc(&tiles, h_tiles.size() * 4));
  HIP_OK(hipMemcpy(rows, h_rows.data(), S * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(bounds, h_bounds.data(), (E + 1) * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(tiles, h_tiles.data(), n_tiles * 4, hipMemcpyHostToDevice));

  {
    std::vector<uint8_t> buf(std::max(gate_bytes, down_bytes));
    for (uint32_t e = 0; e < E; e++) {
      buf.resize(gate_bytes);
      fill_expert(buf, gi, rng);
      HIP_OK(hipMemcpy(static_cast<uint8_t*>(w_gate) + e * gate_bytes, buf.data(), gate_bytes, hipMemcpyHostToDevice));
      fill_expert(buf, gi, rng);
      HIP_OK(hipMemcpy(static_cast<uint8_t*>(w_up) + e * gate_bytes, buf.data(), gate_bytes, hipMemcpyHostToDevice));
      buf.resize(down_bytes);
      fill_expert(buf, di, rng);
      HIP_OK(hipMemcpy(static_cast<uint8_t*>(w_down) + e * down_bytes, buf.data(), down_bytes, hipMemcpyHostToDevice));
    }
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::vector<uint16_t> hx(size_t(S) * K);
    for (auto& v : hx) v = half_bits(nd(rng));
    HIP_OK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    std::vector<float> rw(S, 0.25f);
    HIP_OK(hipMemcpy(route_w, rw.data(), S * 4, hipMemcpyHostToDevice));
  }
  HIP_OK(hipMemset(y16, 0, size_t(S) * M * 2));
  HIP_OK(hipMemset(y32, 0, size_t(S) * K * 4));

  auto gate_up = [&]() {
    int rc = ds41_v41_routed_f16(type, w_gate, w_up, gate_bytes, x, 1, rows, bounds, tiles, n_tiles, route_w, 0.f, y16,
                                 1, M, K);
    if (rc) { std::fprintf(stderr, "routed_f16 gate+up returned %d\n", rc); std::exit(1); }
  };
  auto down = [&]() {
    int rc = ds41_v41_routed_f16(down_type, w_down, nullptr, down_bytes, y16, 1, rows, bounds, tiles, n_tiles, nullptr,
                                 0.f, y32, 0, K, M);
    if (rc) { std::fprintf(stderr, "routed_f16 down returned %d\n", rc); std::exit(1); }
  };

  gate_up();
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  down();
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());

  const float ms_gu = time_ms(gate_up, reps);
  HIP_OK(hipGetLastError());
  const float ms_dn = time_ms(down, reps);
  HIP_OK(hipGetLastError());
  const double flop_proj = 2.0 * S * double(K) * double(M);
  std::printf("gufo gate+up type %d: %u slots over %u experts (mean %.1f rows): %.3f ms, %.1f TFLOPS\n", type, S, E,
              double(S) / E, ms_gu, 2 * flop_proj / (ms_gu * 1e-3) / 1e12);
  std::printf("gufo down type %d: %.3f ms, %.1f TFLOPS\n", down_type, ms_dn, flop_proj / (ms_dn * 1e-3) / 1e12);

  for (void* p : {w_gate, w_up, w_down, (void*)x, (void*)y16, (void*)y32, (void*)route_w, (void*)rows, (void*)bounds,
                  (void*)tiles})
    HIP_OK(hipFree(p));
  return 0;
}

// unholy-probe: what this machine's devices and links can actually do.
//
// Everything the planner relies on is measured here, and every transfer method is verified
// (pattern written on the source, checked on the destination by a kernel) before its bandwidth
// counts: on this box hipMemcpyPeer has been seen to complete without moving data.
//
//   unholy-probe [--json out.json] [--mb N] [--reps N] [--only SECTION[,SECTION...]]
//
// Sections: devices, host, peer, peer-memcpy, zerocopy, signal. `peer-memcpy` (the hipMemcpyPeer
// methods) runs last because one async peer copy has been seen to break later sync ones; run it
// alone (`--only peer-memcpy`) to judge it without that interaction.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "hw/pci.hpp"

#define HIP_CHECK(expr)                                                                  \
  do {                                                                                   \
    hipError_t e_ = (expr);                                                              \
    if (e_ != hipSuccess) {                                                              \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,               \
                   hipGetErrorString(e_));                                               \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

// Soft check: report and return false instead of exiting (for methods that may be unsupported).
#define HIP_TRY(expr)                                                                    \
  ([&]() -> bool {                                                                       \
    hipError_t e_ = (expr);                                                              \
    if (e_ != hipSuccess) {                                                              \
      std::fprintf(stderr, "  [unsupported] %s -> %s\n", #expr, hipGetErrorString(e_));  \
      (void)hipGetLastError();                                                           \
      return false;                                                                      \
    }                                                                                    \
    return true;                                                                         \
  }())

using unholy::hw::PciPath;

namespace {

// ---------------------------------------------------------------------------------------------
// Kernels

__host__ __device__ inline uint32_t pattern(uint32_t seed, uint64_t i) {
  uint32_t x = seed ^ (static_cast<uint32_t>(i) * 0x9E3779B1u) ^ static_cast<uint32_t>(i >> 32);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__global__ void fill_kernel(uint32_t* p, uint64_t n, uint32_t seed) {
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < n;
       i += uint64_t(gridDim.x) * blockDim.x)
    p[i] = pattern(seed, i);
}

__global__ void check_kernel(const uint32_t* p, uint64_t n, uint32_t seed,
                             unsigned long long* bad) {
  unsigned long long local = 0;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < n;
       i += uint64_t(gridDim.x) * blockDim.x)
    local += p[i] != pattern(seed, i);
  if (local) atomicAdd(bad, local);
}

// Vectorized grid-stride copy; with peer/host pointers this is a "pull" or "push" over the link.
__global__ void copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, uint64_t n) {
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < n;
       i += uint64_t(gridDim.x) * blockDim.x)
    dst[i] = src[i];
  __threadfence_system();
}

// Read bandwidth: xor everything, one store per thread so the loads cannot be elided.
__global__ void read_kernel(const uint4* __restrict__ src, uint64_t n, uint32_t* sink) {
  uint32_t acc = 0;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < n;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const uint4 v = src[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

__global__ void tiny_kernel(uint32_t* p) {
  if (threadIdx.x == 0) p[0] += 1;
}

// Persistent ping-pong. `out` is written by this side, `in` polled. Both are system-coherent.
// Returns elapsed wall-clock ticks in result[0], completed iterations in result[1].
__device__ inline uint32_t sys_load(const uint32_t* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline void sys_store(uint32_t* p, uint32_t v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void pingpong_kernel(uint32_t* out, const uint32_t* in, uint32_t iters, bool initiator,
                                uint64_t timeout_ticks, unsigned long long* result) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const uint64_t t0 = wall_clock64();
  uint32_t done = 0;
  for (uint32_t i = 1; i <= iters; ++i) {
    if (initiator) sys_store(out, i);
    while (sys_load(in) < i) {
      if (wall_clock64() - t0 > timeout_ticks) goto out_label;
    }
    if (!initiator) sys_store(out, i);
    done = i;
  }
out_label:
  result[0] = wall_clock64() - t0;
  result[1] = done;
}

// ---------------------------------------------------------------------------------------------
// Host side

struct Device {
  int id = 0;
  hipDeviceProp_t prop{};
  std::string bdf;
  std::string arch;
  bool integrated = false;
  bool stream_wait_value = false;
  int wall_clock_khz = 0;
  std::optional<PciPath> path;
};

struct Opts {
  std::string json_path;
  size_t mb = 256;
  int reps = 5;
  std::set<std::string> only;
  bool want(const char* s) const { return only.empty() || only.count(s); }
};

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

std::string fmt_gbs(double bytes_per_s) {
  char b[32];
  std::snprintf(b, sizeof b, "%.2f GB/s", bytes_per_s / 1e9);
  return b;
}

// Minimal JSON accumulation: the probe emits flat records; the planner parses them.
struct Json {
  std::vector<std::string> devices, links, transfers, signals;

  static std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
      if (c == '"' || c == '\\') o += '\\';
      o += c;
    }
    return o;
  }
  void write(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
      std::perror(path.c_str());
      return;
    }
    auto arr = [&](const char* name, const std::vector<std::string>& v, bool last) {
      std::fprintf(f, "  \"%s\": [\n", name);
      for (size_t i = 0; i < v.size(); ++i)
        std::fprintf(f, "    %s%s\n", v[i].c_str(), i + 1 < v.size() ? "," : "");
      std::fprintf(f, "  ]%s\n", last ? "" : ",");
    };
    std::fprintf(f, "{\n  \"schema\": \"unholy-probe/1\",\n");
    arr("devices", devices, false);
    arr("links", links, false);
    arr("transfers", transfers, false);
    arr("signals", signals, true);
    std::fprintf(f, "}\n");
    std::fclose(f);
  }
};

Json g_json;

void record_transfer(const std::string& method, int src, int dst, size_t bytes, double bytes_per_s,
                     unsigned long long bad, bool concurrent = false, const std::string& note = "") {
  std::ostringstream o;
  o << "{\"method\": \"" << method << "\", \"src\": " << src << ", \"dst\": " << dst
    << ", \"bytes\": " << bytes << ", \"bytes_per_s\": " << static_cast<uint64_t>(bytes_per_s)
    << ", \"verified\": " << (bad == 0 ? "true" : "false") << ", \"mismatches\": " << bad
    << ", \"concurrent\": " << (concurrent ? "true" : "false");
  if (!note.empty()) o << ", \"note\": \"" << Json::esc(note) << "\"";
  o << "}";
  g_json.transfers.push_back(o.str());
  std::printf("  %-22s %2d -> %-2d %12s  %s%s%s\n", method.c_str(), src, dst,
              fmt_gbs(bytes_per_s).c_str(), bad == 0 ? "verified" : "MISMATCH ",
              bad == 0 ? "" : std::to_string(bad).c_str(), note.empty() ? "" : ("  " + note).c_str());
}

// Device ids: -1 = host memory.
constexpr int kHost = -1;

void fill(int dev, uint32_t* p, size_t n, uint32_t seed, hipStream_t s) {
  HIP_CHECK(hipSetDevice(dev));
  fill_kernel<<<1024, 256, 0, s>>>(p, n, seed);
  HIP_CHECK(hipGetLastError());
}

// Verify `p` (resident on or accessible from `dev`) holds `pattern(seed, i)`.
unsigned long long check_on(int dev, const uint32_t* p, size_t n, uint32_t seed) {
  HIP_CHECK(hipSetDevice(dev));
  unsigned long long* bad = nullptr;
  HIP_CHECK(hipMalloc(&bad, sizeof *bad));
  HIP_CHECK(hipMemset(bad, 0, sizeof *bad));
  check_kernel<<<1024, 256>>>(p, n, seed, bad);
  HIP_CHECK(hipGetLastError());
  unsigned long long h = 0;
  HIP_CHECK(hipMemcpy(&h, bad, sizeof h, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(bad));
  return h;
}

unsigned long long check_host(const uint32_t* p, size_t n, uint32_t seed) {
  unsigned long long bad = 0;
  for (size_t i = 0; i < n; ++i) bad += p[i] != pattern(seed, i);
  return bad;
}

std::vector<Device> enumerate() {
  int n = 0;
  HIP_CHECK(hipGetDeviceCount(&n));
  std::vector<Device> devs;
  for (int i = 0; i < n; ++i) {
    Device d;
    d.id = i;
    HIP_CHECK(hipGetDeviceProperties(&d.prop, i));
    char bdf[32];
    std::snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.0", d.prop.pciDomainID, d.prop.pciBusID,
                  d.prop.pciDeviceID);
    d.bdf = bdf;
    d.arch = d.prop.gcnArchName;
    if (auto c = d.arch.find(':'); c != std::string::npos) d.arch.resize(c);
    d.integrated = d.prop.integrated != 0;
    int v = 0;
    if (hipDeviceGetAttribute(&v, hipDeviceAttributeCanUseStreamWaitValue, i) == hipSuccess)
      d.stream_wait_value = v != 0;
    (void)hipGetLastError();
    if (hipDeviceGetAttribute(&v, hipDeviceAttributeWallClockRate, i) == hipSuccess)
      d.wall_clock_khz = v;
    (void)hipGetLastError();
    d.path = unholy::hw::pci_path(d.bdf);
    devs.push_back(d);
  }
  return devs;
}

void section_devices(const std::vector<Device>& devs) {
  std::printf("== devices\n");
  for (const auto& d : devs) {
    const double link = d.path ? d.path->bottleneck_bytes_per_s() : 0;
    std::printf("  [%d] %-8s %s  %s  %.1f GiB  %d CUs  %s  link %s  waitValue=%d  wallclock=%d kHz\n",
                d.id, d.arch.c_str(), d.bdf.c_str(), d.prop.name, d.prop.totalGlobalMem / 1073741824.0,
                d.prop.multiProcessorCount, d.integrated ? "integrated" : "discrete",
                fmt_gbs(link).c_str(), d.stream_wait_value, d.wall_clock_khz);
    if (d.path) {
      for (const auto& h : d.path->hops)
        std::printf("        %s class=%06x max %.1f GT/s x%d  cur %.1f GT/s x%d\n", h.bdf.c_str(),
                    h.class_code, h.max_gts, h.max_width, h.cur_gts, h.cur_width);
    }
    std::ostringstream o;
    o << "{\"id\": " << d.id << ", \"arch\": \"" << d.arch << "\", \"bdf\": \"" << d.bdf
      << "\", \"name\": \"" << Json::esc(d.prop.name) << "\", \"vram_bytes\": "
      << d.prop.totalGlobalMem << ", \"cus\": " << d.prop.multiProcessorCount
      << ", \"integrated\": " << (d.integrated ? "true" : "false")
      << ", \"stream_wait_value\": " << (d.stream_wait_value ? "true" : "false")
      << ", \"link_bottleneck_bytes_per_s\": " << static_cast<uint64_t>(link) << ", \"path\": [";
    if (d.path)
      for (size_t i = 0; i < d.path->hops.size(); ++i) {
        const auto& h = d.path->hops[i];
        o << (i ? ", " : "") << "{\"bdf\": \"" << h.bdf << "\", \"class\": " << h.class_code
          << ", \"max_gts\": " << h.max_gts << ", \"max_width\": " << h.max_width << "}";
      }
    o << "]}";
    g_json.devices.push_back(o.str());
  }
  // Pairwise path joins and peer capability.
  for (size_t a = 0; a < devs.size(); ++a)
    for (size_t b = a + 1; b < devs.size(); ++b) {
      int ab = 0, ba = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&ab, devs[a].id, devs[b].id));
      HIP_CHECK(hipDeviceCanAccessPeer(&ba, devs[b].id, devs[a].id));
      unholy::hw::PathJoin j;
      if (devs[a].path && devs[b].path) j = unholy::hw::join_paths(*devs[a].path, *devs[b].path);
      std::printf("  pair %d-%d: %s%s  canAccessPeer %d->%d=%d %d->%d=%d\n", devs[a].id, devs[b].id,
                  j.via_root_complex ? "via root complex" : "below switch ",
                  j.via_root_complex ? "" : j.common_bdf.c_str(), devs[a].id, devs[b].id, ab,
                  devs[b].id, devs[a].id, ba);
      std::ostringstream o;
      o << "{\"a\": " << devs[a].id << ", \"b\": " << devs[b].id << ", \"via_root_complex\": "
        << (j.via_root_complex ? "true" : "false") << ", \"common_bdf\": \"" << j.common_bdf
        << "\", \"peer_a_to_b\": " << ab << ", \"peer_b_to_a\": " << ba << "}";
      g_json.links.push_back(o.str());
    }
}

// Host <-> device over pinned memory, each device alone, then all discrete cards at once
// (behind a shared uplink the aggregate stays at one link's worth; on separate root ports it adds).
void section_host(const std::vector<Device>& devs, const Opts& opt) {
  std::printf("== host <-> device (pinned, %zu MiB x %d)\n", opt.mb, opt.reps);
  const size_t bytes = opt.mb << 20, n = bytes / 4;
  struct Buf {
    uint32_t* host;
    uint32_t* dev;
    hipStream_t s;
  };
  std::vector<Buf> bufs(devs.size());
  for (size_t k = 0; k < devs.size(); ++k) {
    HIP_CHECK(hipSetDevice(devs[k].id));
    HIP_CHECK(hipHostMalloc(&bufs[k].host, bytes, hipHostMallocPortable | hipHostMallocMapped));
    HIP_CHECK(hipMalloc(&bufs[k].dev, bytes));
    HIP_CHECK(hipStreamCreateWithFlags(&bufs[k].s, hipStreamNonBlocking));
  }
  for (size_t k = 0; k < devs.size(); ++k) {
    const int id = devs[k].id;
    std::vector<double> h2d, d2h;
    unsigned long long bad_h2d = 0, bad_d2h = 0;
    for (int r = 0; r < opt.reps; ++r) {
      const uint32_t seed = 0x1000u + 97u * r + id;
      for (size_t i = 0; i < n; ++i) bufs[k].host[i] = pattern(seed, i);
      HIP_CHECK(hipSetDevice(id));
      double t0 = now_s();
      HIP_CHECK(hipMemcpyAsync(bufs[k].dev, bufs[k].host, bytes, hipMemcpyHostToDevice, bufs[k].s));
      HIP_CHECK(hipStreamSynchronize(bufs[k].s));
      h2d.push_back(bytes / (now_s() - t0));
      bad_h2d += check_on(id, bufs[k].dev, n, seed);

      const uint32_t seed2 = seed ^ 0xabcdefu;
      fill(id, bufs[k].dev, n, seed2, bufs[k].s);
      HIP_CHECK(hipStreamSynchronize(bufs[k].s));
      t0 = now_s();
      HIP_CHECK(hipMemcpyAsync(bufs[k].host, bufs[k].dev, bytes, hipMemcpyDeviceToHost, bufs[k].s));
      HIP_CHECK(hipStreamSynchronize(bufs[k].s));
      d2h.push_back(bytes / (now_s() - t0));
      bad_d2h += check_host(bufs[k].host, n, seed2);
    }
    record_transfer("h2d", kHost, id, bytes, median(h2d), bad_h2d);
    record_transfer("d2h", id, kHost, bytes, median(d2h), bad_d2h);
  }

  // All discrete devices at once.
  std::vector<size_t> cards;
  for (size_t k = 0; k < devs.size(); ++k)
    if (!devs[k].integrated) cards.push_back(k);
  if (cards.size() < 2) return;
  for (const char* dir : {"h2d", "d2h"}) {
    const bool up = std::strcmp(dir, "h2d") == 0;
    std::vector<double> agg;
    for (int r = 0; r < opt.reps; ++r) {
      for (size_t k : cards) {
        HIP_CHECK(hipSetDevice(devs[k].id));
        HIP_CHECK(hipDeviceSynchronize());
      }
      const double t0 = now_s();
      for (size_t k : cards) {
        HIP_CHECK(hipSetDevice(devs[k].id));
        HIP_CHECK(hipMemcpyAsync(up ? (void*)bufs[k].dev : (void*)bufs[k].host,
                                 up ? (void*)bufs[k].host : (void*)bufs[k].dev, bytes,
                                 up ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost, bufs[k].s));
      }
      for (size_t k : cards) HIP_CHECK(hipStreamSynchronize(bufs[k].s));
      agg.push_back(bytes * cards.size() / (now_s() - t0));
    }
    record_transfer(std::string(dir) + "_all_cards", up ? kHost : -2, up ? -2 : kHost,
                    bytes * cards.size(), median(agg), 0, true, "aggregate over all discrete cards");
  }
  for (auto& b : bufs) {
    HIP_CHECK(hipHostFree(b.host));
    HIP_CHECK(hipFree(b.dev));
    HIP_CHECK(hipStreamDestroy(b.s));
  }
}

void enable_peer(int dev, int peer) {
  HIP_CHECK(hipSetDevice(dev));
  hipError_t e = hipDeviceEnablePeerAccess(peer, 0);
  if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
    std::fprintf(stderr, "  enable peer %d->%d: %s\n", dev, peer, hipGetErrorString(e));
  (void)hipGetLastError();
}

// Device <-> device by every method that does not go through hipMemcpyPeer.
void section_peer(const std::vector<Device>& devs, const Opts& opt) {
  std::printf("== device <-> device (%zu MiB x %d)\n", opt.mb, opt.reps);
  const size_t bytes = opt.mb << 20, n = bytes / 4, n16 = bytes / 16;
  std::map<int, uint32_t*> src, dst;
  std::map<int, hipStream_t> stream;
  for (const auto& d : devs) {
    HIP_CHECK(hipSetDevice(d.id));
    HIP_CHECK(hipMalloc(&src[d.id], bytes));
    HIP_CHECK(hipMalloc(&dst[d.id], bytes));
    HIP_CHECK(hipStreamCreateWithFlags(&stream[d.id], hipStreamNonBlocking));
  }
  // Staging slots for the host-staged pipeline (chunked, double-buffered).
  const size_t chunk = std::min<size_t>(bytes, 16u << 20);
  uint8_t* stage[2];
  for (auto& p : stage) HIP_CHECK(hipHostMalloc(&p, chunk, hipHostMallocPortable | hipHostMallocMapped));

  for (const auto& a : devs)
    for (const auto& b : devs) {
      if (a.id == b.id) continue;
      int can_pull = 0, can_push = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_pull, b.id, a.id));  // b reads a
      HIP_CHECK(hipDeviceCanAccessPeer(&can_push, a.id, b.id));  // a writes b
      if (can_pull) enable_peer(b.id, a.id);
      if (can_push) enable_peer(a.id, b.id);

      // pull: kernel on b reads a's memory.
      if (can_pull) {
        std::vector<double> bw;
        unsigned long long bad = 0;
        for (int r = 0; r < opt.reps; ++r) {
          const uint32_t seed = 0x2000u + 131u * r + 16u * a.id + b.id;
          fill(a.id, src[a.id], n, seed, stream[a.id]);
          HIP_CHECK(hipStreamSynchronize(stream[a.id]));
          HIP_CHECK(hipSetDevice(b.id));
          const double t0 = now_s();
          copy_kernel<<<b.prop.multiProcessorCount * 4, 256, 0, stream[b.id]>>>(
              reinterpret_cast<const uint4*>(src[a.id]), reinterpret_cast<uint4*>(dst[b.id]), n16);
          HIP_CHECK(hipStreamSynchronize(stream[b.id]));
          bw.push_back(bytes / (now_s() - t0));
          bad += check_on(b.id, dst[b.id], n, seed);
        }
        record_transfer("pull_kernel", a.id, b.id, bytes, median(bw), bad);
      }
      // push: kernel on a writes b's memory; checked on b after a's stream completes.
      if (can_push) {
        std::vector<double> bw;
        unsigned long long bad = 0;
        for (int r = 0; r < opt.reps; ++r) {
          const uint32_t seed = 0x3000u + 131u * r + 16u * a.id + b.id;
          fill(a.id, src[a.id], n, seed, stream[a.id]);
          HIP_CHECK(hipSetDevice(b.id));
          HIP_CHECK(hipMemset(dst[b.id], 0, bytes));
          HIP_CHECK(hipSetDevice(a.id));
          HIP_CHECK(hipStreamSynchronize(stream[a.id]));
          const double t0 = now_s();
          copy_kernel<<<a.prop.multiProcessorCount * 4, 256, 0, stream[a.id]>>>(
              reinterpret_cast<const uint4*>(src[a.id]), reinterpret_cast<uint4*>(dst[b.id]), n16);
          HIP_CHECK(hipStreamSynchronize(stream[a.id]));
          bw.push_back(bytes / (now_s() - t0));
          bad += check_on(b.id, dst[b.id], n, seed);
        }
        record_transfer("push_kernel", a.id, b.id, bytes, median(bw), bad);
      }
      // host-staged pipeline: d2h chunk on a -> pinned slot -> h2d chunk on b, two slots in flight,
      // cross-device ordering by events only (no host sync inside the loop).
      {
        std::vector<double> bw;
        unsigned long long bad = 0;
        hipEvent_t down[2], up[2];
        for (int k = 0; k < 2; ++k) {
          HIP_CHECK(hipSetDevice(a.id));
          HIP_CHECK(hipEventCreateWithFlags(&down[k], hipEventDisableTiming));
          HIP_CHECK(hipSetDevice(b.id));
          HIP_CHECK(hipEventCreateWithFlags(&up[k], hipEventDisableTiming));
        }
        for (int r = 0; r < opt.reps; ++r) {
          const uint32_t seed = 0x4000u + 131u * r + 16u * a.id + b.id;
          fill(a.id, src[a.id], n, seed, stream[a.id]);
          HIP_CHECK(hipStreamSynchronize(stream[a.id]));
          HIP_CHECK(hipSetDevice(b.id));
          HIP_CHECK(hipMemset(dst[b.id], 0, bytes));
          const double t0 = now_s();
          const size_t nchunks = (bytes + chunk - 1) / chunk;
          for (size_t c = 0; c < nchunks; ++c) {
            const int slot = c & 1;
            const size_t off = c * chunk, len = std::min(chunk, bytes - off);
            HIP_CHECK(hipSetDevice(a.id));
            if (c >= 2) HIP_CHECK(hipStreamWaitEvent(stream[a.id], up[slot], 0));
            HIP_CHECK(hipMemcpyAsync(stage[slot], reinterpret_cast<uint8_t*>(src[a.id]) + off, len,
                                     hipMemcpyDeviceToHost, stream[a.id]));
            HIP_CHECK(hipEventRecord(down[slot], stream[a.id]));
            HIP_CHECK(hipSetDevice(b.id));
            HIP_CHECK(hipStreamWaitEvent(stream[b.id], down[slot], 0));
            HIP_CHECK(hipMemcpyAsync(reinterpret_cast<uint8_t*>(dst[b.id]) + off, stage[slot], len,
                                     hipMemcpyHostToDevice, stream[b.id]));
            HIP_CHECK(hipEventRecord(up[slot], stream[b.id]));
          }
          HIP_CHECK(hipSetDevice(b.id));
          HIP_CHECK(hipStreamSynchronize(stream[b.id]));
          bw.push_back(bytes / (now_s() - t0));
          bad += check_on(b.id, dst[b.id], n, seed);
        }
        for (int k = 0; k < 2; ++k) {
          HIP_CHECK(hipEventDestroy(down[k]));
          HIP_CHECK(hipEventDestroy(up[k]));
        }
        record_transfer("host_staged", a.id, b.id, bytes, median(bw), bad);
      }
    }
  for (auto& p : stage) HIP_CHECK(hipHostFree(p));
  for (const auto& d : devs) {
    HIP_CHECK(hipSetDevice(d.id));
    HIP_CHECK(hipFree(src[d.id]));
    HIP_CHECK(hipFree(dst[d.id]));
    HIP_CHECK(hipStreamDestroy(stream[d.id]));
  }
}

// hipMemcpyPeer / hipMemcpyPeerAsync, verified the way a caller would rely on them: the check
// kernel runs on the destination right after the call returns (sync) or the source stream
// completes (async), with no extra device-wide synchronization.
void section_peer_memcpy(const std::vector<Device>& devs, const Opts& opt) {
  std::printf("== hipMemcpyPeer (%zu MiB x %d)\n", opt.mb, opt.reps);
  const size_t bytes = opt.mb << 20, n = bytes / 4;
  std::map<int, uint32_t*> src, dst;
  std::map<int, hipStream_t> stream;
  for (const auto& d : devs) {
    HIP_CHECK(hipSetDevice(d.id));
    HIP_CHECK(hipMalloc(&src[d.id], bytes));
    HIP_CHECK(hipMalloc(&dst[d.id], bytes));
    HIP_CHECK(hipStreamCreateWithFlags(&stream[d.id], hipStreamNonBlocking));
  }
  for (const char* method : {"memcpy_peer_sync", "memcpy_peer_async"}) {
    const bool async = std::strcmp(method, "memcpy_peer_async") == 0;
    for (const auto& a : devs)
      for (const auto& b : devs) {
        if (a.id == b.id) continue;
        std::vector<double> bw;
        unsigned long long bad = 0;
        for (int r = 0; r < opt.reps; ++r) {
          const uint32_t seed = (async ? 0x6000u : 0x5000u) + 131u * r + 16u * a.id + b.id;
          fill(a.id, src[a.id], n, seed, stream[a.id]);
          HIP_CHECK(hipStreamSynchronize(stream[a.id]));
          HIP_CHECK(hipSetDevice(b.id));
          HIP_CHECK(hipMemset(dst[b.id], 0, bytes));
          HIP_CHECK(hipDeviceSynchronize());
          HIP_CHECK(hipSetDevice(a.id));
          const double t0 = now_s();
          if (async) {
            HIP_CHECK(hipMemcpyPeerAsync(dst[b.id], b.id, src[a.id], a.id, bytes, stream[a.id]));
            HIP_CHECK(hipStreamSynchronize(stream[a.id]));
          } else {
            HIP_CHECK(hipMemcpyPeer(dst[b.id], b.id, src[a.id], a.id, bytes));
          }
          bw.push_back(bytes / (now_s() - t0));
          bad += check_on(b.id, dst[b.id], n, seed);
        }
        record_transfer(method, a.id, b.id, bytes, median(bw), bad);
      }
  }
  for (const auto& d : devs) {
    HIP_CHECK(hipSetDevice(d.id));
    HIP_CHECK(hipFree(src[d.id]));
    HIP_CHECK(hipFree(dst[d.id]));
    HIP_CHECK(hipStreamDestroy(stream[d.id]));
  }
}

// Kernel reads: each device's own memory, and mapped pinned host memory (zero-copy).
void section_zerocopy(const std::vector<Device>& devs, const Opts& opt) {
  std::printf("== kernel read bandwidth (%zu MiB)\n", opt.mb);
  const size_t bytes = opt.mb << 20, n16 = bytes / 16;
  uint8_t* host = nullptr;
  HIP_CHECK(hipHostMalloc(&host, bytes, hipHostMallocPortable | hipHostMallocMapped));
  std::memset(host, 1, bytes);
  for (const auto& d : devs) {
    HIP_CHECK(hipSetDevice(d.id));
    uint8_t* local = nullptr;
    uint32_t* sink = nullptr;
    HIP_CHECK(hipMalloc(&local, bytes));
    HIP_CHECK(hipMalloc(&sink, 4));
    HIP_CHECK(hipMemset(local, 1, bytes));
    void* host_dev = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&host_dev, host, 0));
    for (int which = 0; which < 2; ++which) {
      const uint4* p = reinterpret_cast<const uint4*>(which ? host_dev : local);
      std::vector<double> bw;
      for (int r = 0; r < opt.reps + 1; ++r) {
        HIP_CHECK(hipDeviceSynchronize());
        const double t0 = now_s();
        read_kernel<<<d.prop.multiProcessorCount * 8, 256>>>(p, n16, sink);
        HIP_CHECK(hipDeviceSynchronize());
        if (r) bw.push_back(bytes / (now_s() - t0));  // first run warms TLB/mapping
      }
      record_transfer(which ? "kernel_read_host" : "kernel_read_local", which ? kHost : d.id, d.id,
                      bytes, median(bw), 0, false, "read-only, unverified by design");
    }
    HIP_CHECK(hipFree(local));
    HIP_CHECK(hipFree(sink));
  }
  HIP_CHECK(hipHostFree(host));
}

void record_signal(const std::string& method, int a, int b, int iters, double rt_s, bool ok,
                   const std::string& note = "") {
  std::ostringstream o;
  o << "{\"method\": \"" << method << "\", \"a\": " << a << ", \"b\": " << b
    << ", \"iters\": " << iters << ", \"round_trip_us\": " << rt_s * 1e6
    << ", \"ok\": " << (ok ? "true" : "false");
  if (!note.empty()) o << ", \"note\": \"" << Json::esc(note) << "\"";
  o << "}";
  g_json.signals.push_back(o.str());
  std::printf("  %-26s %d <-> %d  %9.2f us/round trip  %s%s%s\n", method.c_str(), a, b, rt_s * 1e6,
              ok ? "ok" : "FAILED", note.empty() ? "" : "  ", note.c_str());
}

// Round trip a -> b -> a, the shape of one decode-layer hop to the APU and back.
void section_signal(const std::vector<Device>& devs, const Opts& opt) {
  std::printf("== signalling round trip\n");
  const int iters = 2000;
  for (size_t ia = 0; ia < devs.size(); ++ia)
    for (size_t ib = ia + 1; ib < devs.size(); ++ib) {
      const Device& A = devs[ia];
      const Device& B = devs[ib];
      hipStream_t sa, sb;
      uint32_t *ka, *kb;
      HIP_CHECK(hipSetDevice(A.id));
      HIP_CHECK(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
      HIP_CHECK(hipMalloc(&ka, 4));
      HIP_CHECK(hipSetDevice(B.id));
      HIP_CHECK(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
      HIP_CHECK(hipMalloc(&kb, 4));

      // 1. Host-synchronized: kernel, wait on host, kernel on the other side, wait.
      {
        const int it = 200;
        const double t0 = now_s();
        for (int i = 0; i < it; ++i) {
          HIP_CHECK(hipSetDevice(A.id));
          tiny_kernel<<<1, 32, 0, sa>>>(ka);
          HIP_CHECK(hipStreamSynchronize(sa));
          HIP_CHECK(hipSetDevice(B.id));
          tiny_kernel<<<1, 32, 0, sb>>>(kb);
          HIP_CHECK(hipStreamSynchronize(sb));
        }
        record_signal("host_sync", A.id, B.id, it, (now_s() - t0) / it, true);
      }
      // 2. Cross-device events: everything enqueued up front, ordering by hipStreamWaitEvent.
      {
        hipEvent_t ea, eb;
        HIP_CHECK(hipSetDevice(A.id));
        HIP_CHECK(hipEventCreateWithFlags(&ea, hipEventDisableTiming));
        HIP_CHECK(hipSetDevice(B.id));
        HIP_CHECK(hipEventCreateWithFlags(&eb, hipEventDisableTiming));
        const double t0 = now_s();
        for (int i = 0; i < iters; ++i) {
          HIP_CHECK(hipSetDevice(A.id));
          if (i) HIP_CHECK(hipStreamWaitEvent(sa, eb, 0));
          tiny_kernel<<<1, 32, 0, sa>>>(ka);
          HIP_CHECK(hipEventRecord(ea, sa));
          HIP_CHECK(hipSetDevice(B.id));
          HIP_CHECK(hipStreamWaitEvent(sb, ea, 0));
          tiny_kernel<<<1, 32, 0, sb>>>(kb);
          HIP_CHECK(hipEventRecord(eb, sb));
        }
        HIP_CHECK(hipStreamSynchronize(sb));
        record_signal("event_wait", A.id, B.id, iters, (now_s() - t0) / iters, true,
                      "includes host enqueue cost");
        HIP_CHECK(hipEventDestroy(ea));
        HIP_CHECK(hipEventDestroy(eb));
      }
      // 3. Stream memory ops on coherent host flags (GPU front-end waits, no host involvement).
      if (A.stream_wait_value && B.stream_wait_value) {
        uint32_t* flags = nullptr;
        HIP_CHECK(hipHostMalloc(&flags, 64, hipHostMallocPortable | hipHostMallocMapped |
                                                hipHostMallocCoherent));
        std::memset(flags, 0, 64);
        uint32_t* fa = flags;      // written by A
        uint32_t* fb = flags + 8;  // written by B (separate cache line)
        bool ok = true;
        const int it = 1000;
        const double t0 = now_s();
        for (int i = 1; i <= it && ok; ++i) {
          HIP_CHECK(hipSetDevice(A.id));
          tiny_kernel<<<1, 32, 0, sa>>>(ka);
          ok &= HIP_TRY(hipStreamWriteValue32(sa, fa, i, 0));
          ok &= HIP_TRY(hipStreamWaitValue32(sa, fb, i, hipStreamWaitValueGte, 0xffffffffu));
          HIP_CHECK(hipSetDevice(B.id));
          ok &= HIP_TRY(hipStreamWaitValue32(sb, fa, i, hipStreamWaitValueGte, 0xffffffffu));
          tiny_kernel<<<1, 32, 0, sb>>>(kb);
          ok &= HIP_TRY(hipStreamWriteValue32(sb, fb, i, 0));
        }
        HIP_CHECK(hipSetDevice(A.id));
        HIP_CHECK(hipStreamSynchronize(sa));
        HIP_CHECK(hipSetDevice(B.id));
        HIP_CHECK(hipStreamSynchronize(sb));
        record_signal("stream_wait_value", A.id, B.id, it, (now_s() - t0) / it, ok && fb[0] == it,
                      "includes host enqueue cost");
        HIP_CHECK(hipHostFree(flags));
      } else {
        record_signal("stream_wait_value", A.id, B.id, 0, 0, false, "attribute not supported");
      }
      // 4. Persistent kernels spinning on system-scope atomics: flags in coherent host memory,
      //    and flags in each side's own memory written by the other over the link.
      for (int variant = 0; variant < 2; ++variant) {
        uint32_t *fa = nullptr, *fb = nullptr;  // fa: written by A, polled by B
        uint32_t* host_flags = nullptr;
        if (variant == 0) {
          HIP_CHECK(hipHostMalloc(&host_flags, 256, hipHostMallocPortable | hipHostMallocMapped |
                                                        hipHostMallocCoherent));
          std::memset(host_flags, 0, 256);
          fa = host_flags;
          fb = host_flags + 32;
        } else {
          int ab = 0, ba = 0;
          HIP_CHECK(hipDeviceCanAccessPeer(&ab, A.id, B.id));
          HIP_CHECK(hipDeviceCanAccessPeer(&ba, B.id, A.id));
          if (!ab || !ba) {
            record_signal("persistent_peer_flags", A.id, B.id, 0, 0, false, "no peer access");
            continue;
          }
          enable_peer(A.id, B.id);
          enable_peer(B.id, A.id);
          // fa lives on B (B polls locally, A writes remotely); fb lives on A.
          HIP_CHECK(hipSetDevice(B.id));
          if (!HIP_TRY(hipExtMallocWithFlags(reinterpret_cast<void**>(&fa), 64,
                                             hipDeviceMallocFinegrained))) {
            record_signal("persistent_peer_flags", A.id, B.id, 0, 0, false,
                          "fine-grained device memory unsupported");
            continue;
          }
          HIP_CHECK(hipMemset(fa, 0, 64));
          HIP_CHECK(hipSetDevice(A.id));
          if (!HIP_TRY(hipExtMallocWithFlags(reinterpret_cast<void**>(&fb), 64,
                                             hipDeviceMallocFinegrained))) {
            record_signal("persistent_peer_flags", A.id, B.id, 0, 0, false,
                          "fine-grained device memory unsupported");
            continue;
          }
          HIP_CHECK(hipMemset(fb, 0, 64));
          HIP_CHECK(hipDeviceSynchronize());
        }
        unsigned long long *ra = nullptr, *rb = nullptr;
        HIP_CHECK(hipHostMalloc(&ra, 16, hipHostMallocPortable | hipHostMallocMapped));
        HIP_CHECK(hipHostMalloc(&rb, 16, hipHostMallocPortable | hipHostMallocMapped));
        ra[0] = ra[1] = rb[0] = rb[1] = 0;
        const uint64_t timeout_a = uint64_t(A.wall_clock_khz) * 3000;  // 3 s
        const uint64_t timeout_b = uint64_t(B.wall_clock_khz) * 3000;
        HIP_CHECK(hipSetDevice(B.id));
        pingpong_kernel<<<1, 32, 0, sb>>>(fb, fa, iters, false, timeout_b, rb);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipSetDevice(A.id));
        pingpong_kernel<<<1, 32, 0, sa>>>(fa, fb, iters, true, timeout_a, ra);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(sa));
        HIP_CHECK(hipSetDevice(B.id));
        HIP_CHECK(hipStreamSynchronize(sb));
        const bool ok = ra[1] == unsigned(iters) && rb[1] == unsigned(iters);
        const double rt = A.wall_clock_khz ? double(ra[0]) / (A.wall_clock_khz * 1e3) / iters : 0;
        record_signal(variant == 0 ? "persistent_host_flags" : "persistent_peer_flags", A.id, B.id,
                      iters, rt, ok,
                      ok ? "device-timed" : ("completed " + std::to_string(ra[1]) + "/" +
                                             std::to_string(rb[1])));
        HIP_CHECK(hipHostFree(ra));
        HIP_CHECK(hipHostFree(rb));
        if (host_flags) HIP_CHECK(hipHostFree(host_flags));
        if (variant == 1) {
          HIP_CHECK(hipSetDevice(B.id));
          HIP_CHECK(hipFree(fa));
          HIP_CHECK(hipSetDevice(A.id));
          HIP_CHECK(hipFree(fb));
        }
      }
      HIP_CHECK(hipSetDevice(A.id));
      HIP_CHECK(hipStreamDestroy(sa));
      HIP_CHECK(hipFree(ka));
      HIP_CHECK(hipSetDevice(B.id));
      HIP_CHECK(hipStreamDestroy(sb));
      HIP_CHECK(hipFree(kb));
    }
}

}  // namespace

int main(int argc, char** argv) {
  Opts opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--json") opt.json_path = next();
    else if (a == "--mb") opt.mb = std::stoul(next());
    else if (a == "--reps") opt.reps = std::stoi(next());
    else if (a == "--only") {
      std::stringstream ss(next());
      std::string s;
      while (std::getline(ss, s, ',')) opt.only.insert(s);
    } else {
      std::fprintf(stderr, "usage: %s [--json out.json] [--mb N] [--reps N] [--only a,b]\n", argv[0]);
      return 2;
    }
  }

  const auto devs = enumerate();
  if (devs.empty()) {
    std::fprintf(stderr, "no HIP devices\n");
    return 1;
  }
  section_devices(devs);
  if (opt.want("host")) section_host(devs, opt);
  if (opt.want("peer")) section_peer(devs, opt);
  if (opt.want("zerocopy")) section_zerocopy(devs, opt);
  if (opt.want("signal")) section_signal(devs, opt);
  if (opt.want("peer-memcpy")) section_peer_memcpy(devs, opt);
  if (!opt.json_path.empty()) g_json.write(opt.json_path);
  return 0;
}

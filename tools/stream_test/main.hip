// unholy-stream-test: how fast routed experts come off the SSDs into pinned host memory the APU can read, as the
// expert store (docs/PLAN-strix-only.md) will issue them: whole experts in random order, pread from N threads per
// device.
//
//   HIP_VISIBLE_DEVICES=2 unholy-stream-test --pack /stream/dsv4-flash/experts.pack --model /scratch/dsv4-ve
//       [--sources stream,nvme,nvme+repack,both,both+repack] [--qd 1,2,4,8,16] [--gb 20] [--apu-load off,on]
//       [--apu-buf-gb 8] [--device 0] [--seed 1] [--json OUT.json]
//
// Sources
//   stream        the pack on /stream: one slot-sized O_DIRECT read per expert, straight into a pinned slot
//   nvme          the checkpoint on /scratch: six aligned-superset O_DIRECT reads per expert into a pinned stage
//   nvme+repack   the same, then mx_repack_kernel (on the APU) from the stage into a pinned slot, timed in the latency
//   both          stream and nvme at once, one thread group per device; each expert goes to the device whose thread
//                 frees first (a shared work counter); both+repack repacks the nvme experts
// APU load: a bandwidth-bound read kernel over a big buffer on the APU, run alone first and then during every read
// config, so both sides report their numbers with and without the other.
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <mutex>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gufo/json.hpp"
#include "kernels/mx_moe.hpp"
#include "model/expert_pack.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"

using namespace unholy;
using namespace unholy::model;
using namespace unholy::model::pack;
using gufo::json::Value;

namespace {

double now() {
  using C = std::chrono::steady_clock;
  return std::chrono::duration<double>(C::now().time_since_epoch()).count();
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  size_t p = 0;
  while (p <= s.size()) {
    size_t e = s.find(',', p);
    if (e == std::string::npos) e = s.size();
    if (e > p) out.push_back(s.substr(p, e - p));
    p = e + 1;
  }
  return out;
}

// ---- APU bandwidth load ----------------------------------------------------------------------------------------

__global__ void read_bw_kernel(const uint4* __restrict__ p, size_t n, unsigned* sink) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
    acc.x ^= a.x ^ b.x ^ c.x ^ d.x;
    acc.y ^= a.y ^ b.y ^ c.y ^ d.y;
    acc.z ^= a.z ^ b.z ^ c.z ^ d.z;
    acc.w ^= a.w ^ b.w ^ c.w ^ d.w;
  }
  for (; i < n; i += stride) {
    const uint4 a = p[i];
    acc.x ^= a.x;
    acc.y ^= a.y;
    acc.z ^= a.z;
    acc.w ^= a.w;
  }
  const unsigned v = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (v == 0x9e3779b9u) sink[0] = v;  // keeps the loads alive
}

class ApuLoad {
 public:
  ApuLoad(int device, size_t bytes) : device_(device), bytes_(bytes / 16 * 16) {
    UH_CHECK(hipSetDevice(device));
    UH_CHECK(hipMalloc(&buf_, bytes_));
    UH_CHECK(hipMalloc(&sink_, 64));
    UH_CHECK(hipMemset(buf_, 0x5a, bytes_));
    UH_CHECK(hipStreamCreate(&st_));
    run_pass();  // warm
  }
  ~ApuLoad() {
    (void)hipFree(buf_);
    (void)hipFree(sink_);
  }
  size_t bytes() const { return bytes_; }
  // Runs passes on a background thread until stop(); returns achieved GB/s over the completed passes.
  void start() {
    stop_ = false;
    passes_ = 0;
    t_start_ = now();
    t_last_ = t_start_;
    th_ = std::thread([this] {
      UH_CHECK(hipSetDevice(device_));
      while (!stop_) {
        run_pass();
        ++passes_;
        t_last_ = now();
      }
    });
  }
  double stop() {
    stop_ = true;
    th_.join();
    const double dt = t_last_ - t_start_;
    return passes_ ? double(passes_) * double(bytes_) / 1e9 / dt : 0.0;
  }
  double alone(double seconds) {
    start();
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    return stop();
  }

 private:
  void run_pass() {
    read_bw_kernel<<<4096, 256, 0, st_>>>(static_cast<const uint4*>(buf_), bytes_ / 16, static_cast<unsigned*>(sink_));
    UH_CHECK_LAUNCH();
    UH_CHECK(hipStreamSynchronize(st_));
  }
  int device_;
  size_t bytes_;
  void* buf_ = nullptr;
  void* sink_ = nullptr;
  hipStream_t st_ = nullptr;
  std::thread th_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> passes_{0};
  double t_start_ = 0, t_last_ = 0;
};

// ---- the reader ------------------------------------------------------------------------------------------------

enum Dev { kStream = 0, kNvme = 1 };

struct Env {
  const Checkpoint* ck = nullptr;
  Geometry g;
  Header h;
  int pack_fd = -1;
  std::vector<int> shard_fds;
  int device = 0;
};

// Pinned buffers of one reader thread.
struct Pinned {
  uint8_t* slot = nullptr;       // slot-sized destination
  uint8_t* stage[kParts] = {};   // aligned staging of the six checkpoint tensors
  hipStream_t st = nullptr;
};

template <typename T>
T* host_alloc(size_t bytes) {
  void* p = nullptr;
  UH_CHECK(hipHostMalloc(&p, bytes, hipHostMallocPortable));
  return static_cast<T*>(p);
}

struct DevStats {
  std::vector<double> lat;  // seconds per expert, issue to last byte (incl. the repack when asked)
  uint64_t bytes = 0;
  double last_done = 0;
  double repack_s = 0;
};

struct ConfigResult {
  std::string source;
  int qd = 0;
  bool apu_load = false;
  double wall = 0;
  DevStats dev[2];
  double apu_gbs = 0;       // the load kernel while reads ran
  uint64_t n_experts = 0;
};

double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, size_t(q * double(v.size())))];
}

struct Run {
  const Env& env;
  std::vector<size_t> order;  // slot indices, random
  std::atomic<size_t> next{0};
  size_t total = 0;
};

void read_stream(const Env& env, Pinned& p, size_t si) {
  const uint8_t* got = read_aligned(env.pack_fd, env.h.slots_offset + si * env.h.slot_bytes, env.h.slot_bytes, p.slot);
  (void)got;  // slots are 4 KiB aligned: the read lands at the start of p.slot
}

double read_nvme(const Env& env, Pinned& p, size_t si, bool repack) {
  const int group = int(si / size_t(env.g.n_experts)), e = int(si % size_t(env.g.n_experts));
  const TensorInfo* t[kParts];
  expert_tensors(*env.ck, env.g, group, e, t);
  const uint8_t* src[kParts];
  for (int i = 0; i < kParts; ++i)
    src[i] = read_aligned(env.shard_fds[size_t(t[i]->file)], t[i]->offset, t[i]->nbytes, p.stage[i]);
  if (!repack) return 0;
  const double r0 = now();
  const Geometry& g = env.g;
  kernels::mx_repack_to(src[0], src[1], p.slot + g.part_offset[0], p.slot + g.part_offset[1], g.w13_out, g.w13_in, p.st);
  kernels::mx_repack_to(src[2], src[3], p.slot + g.part_offset[2], p.slot + g.part_offset[3], g.w13_out, g.w13_in, p.st);
  kernels::mx_repack_to(src[4], src[5], p.slot + g.part_offset[4], p.slot + g.part_offset[5], g.w2_out, g.w2_in, p.st);
  UH_CHECK(hipStreamSynchronize(p.st));
  return now() - r0;
}

ConfigResult run_config(const Env& env, std::vector<Pinned>& pins_stream, std::vector<Pinned>& pins_nvme,
                        const std::string& source, int qd, double gb, uint64_t seed, ApuLoad* load) {
  const bool use_stream = source == "stream" || source == "both" || source == "both+repack";
  const bool use_nvme = source != "stream";
  const bool repack = source == "nvme+repack" || source == "both+repack";
  ConfigResult res;
  res.source = source;
  res.qd = qd;
  res.apu_load = load != nullptr;

  Run run{env, {}, {}, 0};
  run.total = size_t(std::max(1.0, std::ceil(gb * 1e9 / double(env.h.slot_bytes))));
  std::mt19937_64 rng(seed ^ (uint64_t(qd) << 32) ^ std::hash<std::string>{}(source));
  std::vector<size_t> perm(env.g.n_slots());
  std::iota(perm.begin(), perm.end(), size_t(0));
  while (run.order.size() < run.total) {
    std::shuffle(perm.begin(), perm.end(), rng);
    run.order.insert(run.order.end(), perm.begin(), perm.end());
  }
  run.order.resize(run.total);
  res.n_experts = run.total;

  std::atomic<bool> failed{false};
  std::string err;
  std::mutex mu;
  std::vector<std::thread> pool;
  std::vector<DevStats> stats;  // per thread
  const int nthreads = (use_stream ? qd : 0) + (use_nvme ? qd : 0);
  stats.resize(size_t(nthreads));
  std::vector<int> thread_dev;
  for (int i = 0; i < qd && use_stream; ++i) thread_dev.push_back(kStream);
  for (int i = 0; i < qd && use_nvme; ++i) thread_dev.push_back(kNvme);

  if (load) load->start();
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  double t0 = 0;
  for (int t = 0; t < nthreads; ++t)
    pool.emplace_back([&, t] {
      try {
        UH_CHECK(hipSetDevice(env.device));
        const int d = thread_dev[size_t(t)];
        Pinned& p = d == kStream ? pins_stream[size_t(t)] : pins_nvme[size_t(t) - (use_stream ? size_t(qd) : 0)];
        DevStats& s = stats[size_t(t)];
        ++ready;
        while (!go) std::this_thread::yield();
        for (size_t k; !failed && (k = run.next.fetch_add(1)) < run.total;) {
          const size_t si = run.order[k];
          const double a = now();
          if (d == kStream) read_stream(env, p, si);
          else s.repack_s += read_nvme(env, p, si, repack);
          const double b = now();
          s.lat.push_back(b - a);
          s.bytes += env.h.slot_bytes;
          s.last_done = b;
        }
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu);
        err = e.what();
        failed = true;
      }
    });
  while (ready < nthreads && !failed) std::this_thread::yield();
  t0 = now();
  go = true;
  for (auto& th : pool) th.join();
  const double t1 = now();
  if (load) res.apu_gbs = load->stop();
  if (failed) throw std::runtime_error(err);
  res.wall = t1 - t0;
  for (int t = 0; t < nthreads; ++t) {
    DevStats& d = res.dev[thread_dev[size_t(t)]];
    d.lat.insert(d.lat.end(), stats[size_t(t)].lat.begin(), stats[size_t(t)].lat.end());
    d.bytes += stats[size_t(t)].bytes;
    d.last_done = std::max(d.last_done, stats[size_t(t)].last_done);
    d.repack_s += stats[size_t(t)].repack_s;
  }
  for (auto& d : res.dev) d.last_done = d.last_done ? d.last_done - t0 : 0;
  return res;
}

double gbs(const DevStats& d, double span) { return span > 0 ? double(d.bytes) / 1e9 / span : 0.0; }

Value to_json(const ConfigResult& r, double apu_alone) {
  Value j = Value::object();
  j["source"] = r.source;
  j["queue_depth_per_device"] = r.qd;
  j["apu_load"] = r.apu_load;
  j["experts"] = r.n_experts;
  j["wall_s"] = r.wall;
  uint64_t bytes = r.dev[0].bytes + r.dev[1].bytes;
  j["combined_gbs"] = double(bytes) / 1e9 / r.wall;
  static const char* const names[2] = {"stream", "nvme"};
  std::vector<double> all;
  for (int d = 0; d < 2; ++d) {
    const DevStats& s = r.dev[d];
    if (s.lat.empty()) continue;
    Value o = Value::object();
    o["experts"] = s.lat.size();
    o["gbs"] = gbs(s, s.last_done);
    o["lat_ms_p50"] = pct(s.lat, 0.5) * 1e3;
    o["lat_ms_p90"] = pct(s.lat, 0.9) * 1e3;
    o["lat_ms_p99"] = pct(s.lat, 0.99) * 1e3;
    if (s.repack_s > 0) o["repack_ms_mean"] = s.repack_s / double(s.lat.size()) * 1e3;
    j[names[d]] = std::move(o);
    all.insert(all.end(), s.lat.begin(), s.lat.end());
  }
  j["lat_ms_p50"] = pct(all, 0.5) * 1e3;
  j["lat_ms_p90"] = pct(all, 0.9) * 1e3;
  j["lat_ms_p99"] = pct(all, 0.99) * 1e3;
  if (r.apu_load) {
    j["apu_kernel_gbs"] = r.apu_gbs;
    j["apu_kernel_gbs_alone"] = apu_alone;
  }
  return j;
}

void print_result(const ConfigResult& r, double apu_alone) {
  const uint64_t bytes = r.dev[0].bytes + r.dev[1].bytes;
  std::printf("%-12s qd %2d  %-8s  combined %6.2f GB/s", r.source.c_str(), r.qd, r.apu_load ? "apu-load" : "no-load",
              double(bytes) / 1e9 / r.wall);
  static const char* const names[2] = {"stream", "nvme"};
  for (int d = 0; d < 2; ++d) {
    const DevStats& s = r.dev[d];
    if (s.lat.empty()) continue;
    std::printf("  | %s %5.2f GB/s p50 %6.1f p90 %6.1f p99 %6.1f ms", names[d], gbs(s, s.last_done),
                pct(s.lat, 0.5) * 1e3, pct(s.lat, 0.9) * 1e3, pct(s.lat, 0.99) * 1e3);
    if (s.repack_s > 0) std::printf(" (repack %.2f ms)", s.repack_s / double(s.lat.size()) * 1e3);
  }
  if (r.apu_load) std::printf("  | apu kernel %.1f GB/s (alone %.1f)", r.apu_gbs, apu_alone);
  std::printf("\n");
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  std::string packf = "/stream/dsv4-flash/experts.pack", model = "/scratch/dsv4-ve", json_out;
  std::string sources = "stream,nvme,nvme+repack,both,both+repack", qds = "1,2,4,8,16", loads = "off,on";
  double gb = 20, apu_gb = 8;
  int device = 0;
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--pack") packf = val();
    else if (a == "--model") model = val();
    else if (a == "--sources") sources = val();
    else if (a == "--qd") qds = val();
    else if (a == "--gb") gb = std::stod(val());
    else if (a == "--apu-load") loads = val();
    else if (a == "--apu-buf-gb") apu_gb = std::stod(val());
    else if (a == "--device") device = std::stoi(val());
    else if (a == "--seed") seed = std::stoull(val());
    else if (a == "--json") json_out = val();
    else {
      std::fprintf(stderr, "usage: unholy-stream-test --pack FILE --model DIR [--sources ...] [--qd ...] [--gb N] [--apu-load off,on] "
                           "[--apu-buf-gb N] [--device D] [--seed N] [--json OUT]\n");
      return 2;
    }
  }
  try {
    Env env;
    env.device = device;
    UH_CHECK(hipSetDevice(device));
    hipDeviceProp_t prop;
    UH_CHECK(hipGetDeviceProperties(&prop, device));
    std::printf("device %d: %s\n", device, prop.gcnArchName);
    auto ck = Checkpoint::open(model);
    env.ck = ck.get();
    env.pack_fd = ::open(packf.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (env.pack_fd < 0) throw std::runtime_error("open " + packf + ": " + std::strerror(errno));
    env.h = read_header(env.pack_fd);
    env.g = geometry_from(env.h);
    if (env.h.source_hash != source_hash(*ck, env.g)) throw std::runtime_error("the pack was built from another checkpoint");
    for (const auto& s : ck->shard_paths()) {
      int fd = ::open(s.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
      if (fd < 0) throw std::runtime_error("open " + s + ": " + std::strerror(errno));
      env.shard_fds.push_back(fd);
    }

    std::vector<int> qd_list;
    for (auto& s : split(qds)) qd_list.push_back(std::stoi(s));
    const int max_qd = *std::max_element(qd_list.begin(), qd_list.end());
    std::vector<Pinned> pins_stream, pins_nvme;
    pins_stream.resize(size_t(max_qd));
    pins_nvme.resize(size_t(max_qd));
    for (auto& p : pins_stream) p.slot = host_alloc<uint8_t>(env.h.slot_bytes);
    for (auto& p : pins_nvme) {
      p.slot = host_alloc<uint8_t>(env.h.slot_bytes);
      for (int i = 0; i < kParts; ++i) p.stage[i] = host_alloc<uint8_t>(aligned_buffer_bytes(env.g.part_bytes[i]));
      UH_CHECK(hipStreamCreate(&p.st));
    }

    const bool want_load = sources.size() && loads.find("on") != std::string::npos;
    std::unique_ptr<ApuLoad> load;
    double apu_alone = 0;
    if (want_load) {
      load = std::make_unique<ApuLoad>(device, size_t(apu_gb * 1e9));
      apu_alone = load->alone(4.0);
      std::printf("apu read kernel alone: %.1f GB/s over %.1f GB\n", apu_alone, double(load->bytes()) / 1e9);
    }

    Value results = Value::array();
    std::vector<ConfigResult> all;
    for (const auto& load_mode : split(loads))
      for (const auto& src : split(sources))
        for (int qd : qd_list) {
          const bool with_load = load_mode == "on";
          ConfigResult r = run_config(env, pins_stream, pins_nvme, src, qd, gb, seed, with_load ? load.get() : nullptr);
          print_result(r, apu_alone);
          results.push_back(to_json(r, apu_alone));
          all.push_back(std::move(r));
        }
    double apu_after = 0;
    if (want_load) {
      apu_after = load->alone(4.0);
      std::printf("apu read kernel alone (after): %.1f GB/s\n", apu_after);
    }

    // Best queue depth per source and load mode.
    Value best = Value::array();
    std::printf("\nbest queue depth (combined GB/s):\n");
    for (const auto& load_mode : split(loads))
      for (const auto& src : split(sources)) {
        const ConfigResult* b = nullptr;
        double bg = 0;
        for (const auto& r : all)
          if (r.source == src && r.apu_load == (load_mode == "on")) {
            const double g = double(r.dev[0].bytes + r.dev[1].bytes) / 1e9 / r.wall;
            if (g > bg) bg = g, b = &r;
          }
        if (!b) continue;
        std::printf("  %-12s %-8s qd %2d  %.2f GB/s\n", src.c_str(), load_mode == "on" ? "apu-load" : "no-load", b->qd, bg);
        Value o = Value::object();
        o["source"] = src;
        o["apu_load"] = load_mode == "on";
        o["best_queue_depth"] = b->qd;
        o["combined_gbs"] = bg;
        best.push_back(std::move(o));
      }

    if (!json_out.empty()) {
      Value j = Value::object();
      j["device"] = std::string(prop.gcnArchName);
      j["pack"] = packf;
      j["model"] = model;
      j["gb_per_config"] = gb;
      j["slot_bytes"] = env.h.slot_bytes;
      if (want_load) {
        j["apu_kernel_gbs_alone_before"] = apu_alone;
        j["apu_kernel_gbs_alone_after"] = apu_after;
        j["apu_buffer_gb"] = double(load->bytes()) / 1e9;
      }
      j["best"] = std::move(best);
      j["configs"] = std::move(results);
      FILE* f = std::fopen(json_out.c_str(), "w");
      if (!f) throw std::runtime_error("cannot write " + json_out);
      const std::string s = j.dump();
      std::fwrite(s.data(), 1, s.size(), f);
      std::fputc('\n', f);
      std::fclose(f);
      std::printf("wrote %s\n", json_out.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "unholy-stream-test: %s\n", e.what());
    return 1;
  }
}

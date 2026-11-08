// unholy-expert-pack: builds the expert pack (src/model/expert_pack.hpp) of a native DSV4 checkpoint.
//
//   unholy-expert-pack --model DIR --out FILE [--layers 0,21,m0,5-9] [--threads N] [--verify] [--verify-only]
//
// CPU and disk only: O_DIRECT reads of the checkpoint, repack on the CPU (the mapping of kernels::mx_repack),
// O_DIRECT writes. FILE.json is the readable index. --verify re-reads the pack, checks the slot checksums, and
// inverts the repack to compare every slot with the checkpoint's bytes.
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <memory>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gufo/json.hpp"
#include "model/expert_pack.hpp"

using namespace unholy::model;
using namespace unholy::model::pack;

namespace {

double now() {
  using C = std::chrono::steady_clock;
  return std::chrono::duration<double>(C::now().time_since_epoch()).count();
}

std::string hex(uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof b, "%016llx", (unsigned long long)v);
  return b;
}

struct Shards {
  std::vector<int> fds;
  explicit Shards(const Checkpoint& ck) {
    for (const auto& p : ck.shard_paths()) {
      int fd = ::open(p.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
      if (fd < 0) throw std::runtime_error("open " + p + " (O_DIRECT): " + std::strerror(errno));
      fds.push_back(fd);
    }
  }
  ~Shards() {
    for (int fd : fds) ::close(fd);
  }
};

// Group ids from "0,5-9,m0": plain numbers are main layers, m<i> are the DSpark stages.
std::vector<int> parse_groups(const std::string& spec, const Geometry& g) {
  std::vector<bool> on(size_t(g.n_layers + g.n_mtp), spec.empty());
  size_t i = 0;
  auto num = [&](bool& mtp) {
    mtp = false;
    if (i < spec.size() && spec[i] == 'm') mtp = true, ++i;
    size_t j = i;
    int v = 0;
    while (j < spec.size() && isdigit((unsigned char)spec[j])) v = v * 10 + (spec[j++] - '0');
    if (j == i) throw std::runtime_error("--layers: bad spec " + spec);
    i = j;
    return v;
  };
  while (i < spec.size()) {
    bool m0, m1;
    int a = num(m0), b = a;
    if (i < spec.size() && spec[i] == '-') {
      ++i;
      b = num(m1);
    }
    for (int v = a; v <= b; ++v) {
      const int gi = m0 ? g.n_layers + v : v;
      if (gi < 0 || gi >= g.n_layers + g.n_mtp) throw std::runtime_error("--layers: out of range: " + spec);
      on[size_t(gi)] = true;
    }
    if (i < spec.size()) {
      if (spec[i] != ',') throw std::runtime_error("--layers: bad spec " + spec);
      ++i;
    }
  }
  std::vector<int> out;
  for (size_t k = 0; k < on.size(); ++k)
    if (on[k]) out.push_back(int(k));
  return out;
}

// Runs fn(index) for index in [0, n) on `threads` workers; the first exception stops the others and is rethrown.
template <typename F>
void parallel_for(size_t n, int threads, F fn) {
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr err;
  std::mutex mu;
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      try {
        for (size_t i; !failed && (i = next.fetch_add(1)) < n;) fn(i, t);
      } catch (...) {
        std::lock_guard<std::mutex> lk(mu);
        if (!err) err = std::current_exception();
        failed = true;
      }
    });
  for (auto& th : pool) th.join();
  if (err) std::rethrow_exception(err);
}

// Per-thread buffers: one aligned read buffer per source tensor, the slot.
struct Work {
  uint8_t* rd[kParts];
  uint8_t* slot;
  explicit Work(const Geometry& g) {
    for (int i = 0; i < kParts; ++i) rd[i] = alloc_aligned(aligned_buffer_bytes(g.part_bytes[i]));
    slot = alloc_aligned(g.slot_bytes);
  }
  ~Work() {
    for (auto* p : rd) free_aligned(p);
    free_aligned(slot);
  }
};

struct Plan {
  Geometry g;
  uint64_t slots_offset = kFirstSlotOffset, checksum_offset = 0, total = 0;
};

Plan make_plan(const Checkpoint& ck) {
  Plan p;
  p.g = geometry_of(ck);
  p.checksum_offset = p.slots_offset + p.g.n_slots() * p.g.slot_bytes;
  p.total = p.checksum_offset + align_up(p.g.n_slots() * sizeof(ChecksumEntry));
  return p;
}

void write_all(int fd, const void* buf, size_t len, uint64_t off) {
  size_t done = 0;
  while (done < len) {
    const ssize_t r = ::pwrite(fd, static_cast<const uint8_t*>(buf) + done, len - done, off_t(off + done));
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("pwrite: ") + std::strerror(errno));
    }
    done += size_t(r);
  }
}

void read_all(int fd, void* buf, size_t len, uint64_t off) {
  size_t done = 0;
  while (done < len) {
    const ssize_t r = ::pread(fd, static_cast<uint8_t*>(buf) + done, len - done, off_t(off + done));
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("pread: ") + std::strerror(errno));
    }
    if (r == 0) throw std::runtime_error("pread: EOF");
    done += size_t(r);
  }
}

// Reads the six source tensors of one expert (aligned supersets) into w.rd; returns pointers to the payloads.
void read_expert(const Checkpoint& ck, const Shards& sh, const Geometry& g, int group, int expert, Work& w,
                 const uint8_t* src[kParts]) {
  const TensorInfo* t[kParts];
  expert_tensors(ck, g, group, expert, t);
  for (int i = 0; i < kParts; ++i) src[i] = read_aligned(sh.fds[size_t(t[i]->file)], t[i]->offset, t[i]->nbytes, w.rd[i]);
}

void build_slot(const Geometry& g, const uint8_t* src[kParts], uint8_t* slot) {
  // Parts are in order w1.weight, w1.scale, w3.weight, w3.scale, w2.weight, w2.scale.
  repack_matrix(src[0], src[1], slot + g.part_offset[0], slot + g.part_offset[1], g.w13_out, g.w13_in);
  repack_matrix(src[2], src[3], slot + g.part_offset[2], slot + g.part_offset[3], g.w13_out, g.w13_in);
  repack_matrix(src[4], src[5], slot + g.part_offset[4], slot + g.part_offset[5], g.w2_out, g.w2_in);
}

std::vector<ChecksumEntry> load_table(int fd, const Header& h) {
  std::vector<ChecksumEntry> t(h.n_slots);
  const size_t bytes = align_up(h.n_slots * sizeof(ChecksumEntry));
  uint8_t* buf = alloc_aligned(bytes);
  read_all(fd, buf, bytes, h.checksum_offset);
  std::memcpy(t.data(), buf, t.size() * sizeof(ChecksumEntry));
  free_aligned(buf);
  return t;
}

void write_json(const std::string& path, const Checkpoint& ck, const Header& h, const Geometry& g,
                const std::vector<ChecksumEntry>& table) {
  using gufo::json::Value;
  Value j = Value::object();
  j["format"] = "unholy expert pack";
  j["version"] = int(h.version);
  j["n_layers"] = g.n_layers;
  j["n_mtp"] = g.n_mtp;
  j["n_experts"] = g.n_experts;
  j["slot_bytes"] = h.slot_bytes;
  j["slots_offset"] = h.slots_offset;
  j["checksum_offset"] = h.checksum_offset;
  j["slot_order"] = "group (layers.0..layers.N-1, mtp.0..), then expert; slot i at slots_offset + i * slot_bytes";
  j["checksum"] = "xxh64 (seed 0) of the slot's slot_bytes";
  Value parts = Value::array();
  static const char* const names[kParts] = {"w1.codes", "w1.scales", "w3.codes", "w3.scales", "w2.codes", "w2.scales"};
  for (int i = 0; i < kParts; ++i) {
    Value p = Value::object();
    p["name"] = names[i];
    p["offset"] = h.part_offset[i];
    p["bytes"] = h.part_bytes[i];
    parts.push_back(std::move(p));
  }
  j["parts"] = std::move(parts);
  j["w1_w3_out_in"] = std::to_string(g.w13_out) + "x" + std::to_string(g.w13_in);
  j["w2_out_in"] = std::to_string(g.w2_out) + "x" + std::to_string(g.w2_in);
  j["model_dir"] = ck.dir();
  j["config_hash"] = hex(h.config_hash);
  j["source_hash"] = hex(h.source_hash);
  Value shards = Value::array();
  for (const auto& s : ck.shard_paths()) shards.push_back(std::filesystem::path(s).filename().string());
  j["shards"] = std::move(shards);
  Value slots = Value::array();
  for (int grp = 0; grp < g.n_layers + g.n_mtp; ++grp)
    for (int e = 0; e < g.n_experts; ++e) {
      const size_t si = g.slot_index(grp, e);
      Value s = Value::object();
      s["slot"] = si;
      s["tensors"] = expert_prefix(g, grp, e);
      s["offset"] = h.slots_offset + si * h.slot_bytes;
      s["present"] = table[si].present != 0;
      s["xxh64"] = hex(table[si].hash);
      const TensorInfo* t[kParts];
      expert_tensors(ck, g, grp, e, t);
      Value src = Value::array();
      for (int i = 0; i < kParts; ++i) {
        Value a = Value::array();
        a.push_back(t[i]->file);
        a.push_back(t[i]->offset);
        a.push_back(t[i]->nbytes);
        src.push_back(std::move(a));
      }
      s["source_shard_offset_size"] = std::move(src);
      slots.push_back(std::move(s));
    }
  j["slots"] = std::move(slots);
  std::ofstream f(path);
  f << j.dump() << "\n";
}

int build(const Checkpoint& ck, const std::string& out, const std::string& layers, int threads) {
  const Plan plan = make_plan(ck);
  const Geometry& g = plan.g;
  const auto groups = parse_groups(layers, g);
  Shards sh(ck);

  Header h;
  h.n_layers = uint32_t(g.n_layers);
  h.n_mtp = uint32_t(g.n_mtp);
  h.n_experts = uint32_t(g.n_experts);
  h.n_slots = g.n_slots();
  h.slot_bytes = g.slot_bytes;
  h.slots_offset = plan.slots_offset;
  h.checksum_offset = plan.checksum_offset;
  for (int i = 0; i < kParts; ++i) {
    h.part_offset[i] = g.part_offset[i];
    h.part_bytes[i] = g.part_bytes[i];
  }
  h.w13_out = uint32_t(g.w13_out);
  h.w13_in = uint32_t(g.w13_in);
  h.w2_out = uint32_t(g.w2_out);
  h.w2_in = uint32_t(g.w2_in);
  h.config_hash = config_hash(ck);
  h.source_hash = source_hash(ck, g);

  std::vector<ChecksumEntry> table(g.n_slots(), ChecksumEntry{0, 0});
  int fd = ::open(out.c_str(), O_RDWR | O_CREAT | O_DIRECT | O_CLOEXEC, 0644);
  if (fd < 0) throw std::runtime_error("open " + out + ": " + std::strerror(errno));
  // Keep the slots of an earlier partial build of the same source.
  try {
    const Header old = read_header(fd);
    struct stat st;
    ::fstat(fd, &st);
    if (old.source_hash == h.source_hash && old.config_hash == h.config_hash && old.slot_bytes == h.slot_bytes &&
        old.n_slots == h.n_slots && uint64_t(st.st_size) == plan.total) {
      table = load_table(fd, old);
      std::printf("continuing an earlier pack: %zu slots present\n",
                  size_t(std::count_if(table.begin(), table.end(), [](auto& e) { return e.present; })));
    }
  } catch (const std::exception&) {
  }
  {
    struct stat st;
    ::fstat(fd, &st);
    if (uint64_t(st.st_size) != plan.total) {
      std::fill(table.begin(), table.end(), ChecksumEntry{0, 0});
      if (::ftruncate(fd, 0) || ::ftruncate(fd, off_t(plan.total))) throw std::runtime_error("ftruncate failed");
    }
  }

  std::vector<size_t> todo;  // slot indices
  for (int grp : groups)
    for (int e = 0; e < g.n_experts; ++e) todo.push_back(g.slot_index(grp, e));
  std::vector<std::unique_ptr<Work>> work;
  for (int t = 0; t < threads; ++t) work.push_back(std::make_unique<Work>(g));

  std::atomic<size_t> done{0};
  std::atomic<uint64_t> rd_bytes{0};
  const double t0 = now();
  std::printf("packing %zu slots (%.1f GB) with %d threads\n", todo.size(), double(todo.size() * g.slot_bytes) / 1e9,
              threads);
  parallel_for(todo.size(), threads, [&](size_t k, int t) {
    const size_t si = todo[k];
    const int grp = int(si / size_t(g.n_experts)), e = int(si % size_t(g.n_experts));
    Work& w = *work[size_t(t)];
    const uint8_t* src[kParts];
    read_expert(ck, sh, g, grp, e, w, src);
    build_slot(g, src, w.slot);
    table[si].hash = xxh64(w.slot, g.slot_bytes);
    table[si].present = 1;
    write_all(fd, w.slot, g.slot_bytes, plan.slots_offset + si * g.slot_bytes);
    rd_bytes += g.slot_bytes;
    const size_t d = ++done;
    if (d % 256 == 0 || d == todo.size()) {
      const double dt = now() - t0;
      std::printf("\r%zu / %zu slots, %.2f GB/s written, %.0f s", d, todo.size(), double(d * g.slot_bytes) / 1e9 / dt, dt);
      std::fflush(stdout);
    }
  });
  std::printf("\n");

  // Checksum table, then the header last: a pack with a header is a pack whose table is complete.
  {
    const size_t tb = align_up(table.size() * sizeof(ChecksumEntry));
    uint8_t* buf = alloc_aligned(tb);
    std::memset(buf, 0, tb);
    std::memcpy(buf, table.data(), table.size() * sizeof(ChecksumEntry));
    write_all(fd, buf, tb, plan.checksum_offset);
    free_aligned(buf);
    h.header_hash = xxh64(&h, offsetof(Header, header_hash));
    uint8_t* hb = alloc_aligned(kHeaderBytes);
    std::memset(hb, 0, kHeaderBytes);
    std::memcpy(hb, &h, sizeof h);
    write_all(fd, hb, kHeaderBytes, 0);
    free_aligned(hb);
  }
  ::fsync(fd);
  ::close(fd);
  write_json(out + ".json", ck, h, g, table);
  std::printf("wrote %s (%.1f GB) in %.0f s, index %s.json\n", out.c_str(), double(plan.total) / 1e9, now() - t0,
              out.c_str());
  return 0;
}

int verify(const Checkpoint& ck, const std::string& out, int threads) {
  const Plan plan = make_plan(ck);
  const Geometry& g = plan.g;
  Shards sh(ck);
  int fd = ::open(out.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("open " + out + ": " + std::strerror(errno));
  const Header h = read_header(fd);
  if (h.source_hash != source_hash(ck, g) || h.config_hash != config_hash(ck))
    throw std::runtime_error("verify: the pack was built from a different checkpoint (fingerprint mismatch)");
  if (h.slot_bytes != g.slot_bytes || h.n_slots != g.n_slots() || h.slots_offset != plan.slots_offset ||
      h.checksum_offset != plan.checksum_offset)
    throw std::runtime_error("verify: geometry mismatch");
  const auto table = load_table(fd, h);

  struct VWork {
    Work w;
    uint8_t* back[kParts];
    uint8_t* slot;
    explicit VWork(const Geometry& g) : w(g) {
      for (int i = 0; i < kParts; ++i) back[i] = alloc_aligned(g.part_bytes[i]);
      slot = alloc_aligned(aligned_buffer_bytes(g.slot_bytes));
    }
    ~VWork() {
      for (auto* p : back) free_aligned(p);
      free_aligned(slot);
    }
  };
  std::vector<std::unique_ptr<VWork>> work;
  for (int t = 0; t < threads; ++t) work.push_back(std::make_unique<VWork>(g));
  std::atomic<size_t> checked{0}, bad_hash{0}, bad_bytes{0}, absent{0};
  std::mutex mu;
  const double t0 = now();
  parallel_for(g.n_slots(), threads, [&](size_t si, int t) {
    if (!table[si].present) {
      ++absent;
      return;
    }
    VWork& v = *work[size_t(t)];
    const int grp = int(si / size_t(g.n_experts)), e = int(si % size_t(g.n_experts));
    const uint8_t* slot = read_aligned(fd, h.slots_offset + si * h.slot_bytes, h.slot_bytes, v.slot);
    if (xxh64(slot, h.slot_bytes) != table[si].hash) {
      ++bad_hash;
      std::lock_guard<std::mutex> lk(mu);
      std::fprintf(stderr, "slot %zu (%s): checksum mismatch\n", si, expert_prefix(g, grp, e).c_str());
    }
    unrepack_matrix(slot + g.part_offset[0], slot + g.part_offset[1], v.back[0], v.back[1], g.w13_out, g.w13_in);
    unrepack_matrix(slot + g.part_offset[2], slot + g.part_offset[3], v.back[2], v.back[3], g.w13_out, g.w13_in);
    unrepack_matrix(slot + g.part_offset[4], slot + g.part_offset[5], v.back[4], v.back[5], g.w2_out, g.w2_in);
    const uint8_t* src[kParts];
    read_expert(ck, sh, g, grp, e, v.w, src);
    for (int i = 0; i < kParts; ++i)
      if (std::memcmp(src[i], v.back[i], g.part_bytes[i])) {
        ++bad_bytes;
        std::lock_guard<std::mutex> lk(mu);
        std::fprintf(stderr, "slot %zu (%s): part %d differs from the checkpoint\n", si, expert_prefix(g, grp, e).c_str(), i);
        break;
      }
    const size_t d = ++checked;
    if (d % 256 == 0) {
      std::printf("\rverified %zu slots, %.0f s", d, now() - t0);
      std::fflush(stdout);
    }
  });
  std::printf("\nverify: %zu slots checked (%zu absent), checksum mismatches %zu, byte mismatches %zu, %.0f s\n",
              checked.load(), absent.load(), bad_hash.load(), bad_bytes.load(), now() - t0);
  ::close(fd);
  return bad_hash || bad_bytes ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, out, layers;
  bool do_verify = false, verify_only = false;
  int threads = 16;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--model") model = val();
    else if (a == "--out") out = val();
    else if (a == "--layers") layers = val();
    else if (a == "--threads") threads = std::stoi(val());
    else if (a == "--verify") do_verify = true;
    else if (a == "--verify-only") do_verify = verify_only = true;
    else {
      std::fprintf(stderr,
                   "usage: unholy-expert-pack --model DIR --out FILE [--layers 0,21,m0,5-9] [--threads N] [--verify] "
                   "[--verify-only]\n");
      return 2;
    }
  }
  if (model.empty() || out.empty()) {
    std::fprintf(stderr, "unholy-expert-pack: --model and --out are required\n");
    return 2;
  }
  try {
    auto ck = Checkpoint::open(model);
    int rc = 0;
    if (!verify_only) rc = build(*ck, out, layers, threads);
    if (!rc && do_verify) rc = verify(*ck, out, threads);
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "unholy-expert-pack: %s\n", e.what());
    return 1;
  }
}

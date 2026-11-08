#include "model/expert_pack.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace unholy::model::pack {

namespace {
constexpr uint64_t P1 = 11400714785074694791ull, P2 = 14029467366897019727ull, P3 = 1609587929392839161ull,
                   P4 = 9650029242287828579ull, P5 = 2870177450012600261ull;
inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t rd64(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}
inline uint32_t rd32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
inline uint64_t round1(uint64_t acc, uint64_t in) { return rotl(acc + in * P2, 31) * P1; }
inline uint64_t merge(uint64_t acc, uint64_t v) { return (acc ^ round1(0, v)) * P1 + P4; }
}  // namespace

uint64_t xxh64(const void* data, size_t len, uint64_t seed) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  const uint8_t* end = p + len;
  uint64_t h;
  if (len >= 32) {
    uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
    do {
      v1 = round1(v1, rd64(p));
      v2 = round1(v2, rd64(p + 8));
      v3 = round1(v3, rd64(p + 16));
      v4 = round1(v4, rd64(p + 24));
      p += 32;
    } while (p + 32 <= end);
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    h = merge(h, v1);
    h = merge(h, v2);
    h = merge(h, v3);
    h = merge(h, v4);
  } else {
    h = seed + P5;
  }
  h += len;
  while (p + 8 <= end) {
    h ^= round1(0, rd64(p));
    h = rotl(h, 27) * P1 + P4;
    p += 8;
  }
  if (p + 4 <= end) {
    h ^= uint64_t(rd32(p)) * P1;
    h = rotl(h, 23) * P2 + P3;
    p += 4;
  }
  while (p < end) {
    h ^= uint64_t(*p++) * P5;
    h = rotl(h, 11) * P1;
  }
  h ^= h >> 33;
  h *= P2;
  h ^= h >> 29;
  h *= P3;
  h ^= h >> 32;
  return h;
}

std::string expert_prefix(const Geometry& g, int group, int expert) {
  std::string p = group < g.n_layers ? "layers." + std::to_string(group) : "mtp." + std::to_string(group - g.n_layers);
  return p + ".ffn.experts." + std::to_string(expert) + ".";
}

Geometry geometry_of(const Checkpoint& ck) {
  Geometry g;
  auto count = [&](const std::string& pre) {
    int n = 0;
    while (ck.find(pre + std::to_string(n) + ".ffn.experts.0.w1.weight")) ++n;
    return n;
  };
  g.n_layers = count("layers.");
  g.n_mtp = count("mtp.");
  while (ck.find("layers.0.ffn.experts." + std::to_string(g.n_experts) + ".w1.weight")) ++g.n_experts;
  if (!g.n_layers || !g.n_experts) throw std::runtime_error("expert pack: no routed experts in the checkpoint");
  const auto& w1 = ck.at(expert_prefix(g, 0, 0) + "w1.weight");
  const auto& w2 = ck.at(expert_prefix(g, 0, 0) + "w2.weight");
  g.w13_out = int(w1.shape.at(0));
  g.w13_in = int(w1.shape.at(1)) * 2;
  g.w2_out = int(w2.shape.at(0));
  g.w2_in = int(w2.shape.at(1)) * 2;
  const uint64_t c13 = uint64_t(g.w13_out) * g.w13_in / 2, s13 = uint64_t(g.w13_out) * g.w13_in / 32;
  const uint64_t c2 = uint64_t(g.w2_out) * g.w2_in / 2, s2 = uint64_t(g.w2_out) * g.w2_in / 32;
  const uint64_t sz[kParts] = {c13, s13, c13, s13, c2, s2};
  uint64_t off = 0;
  for (int i = 0; i < kParts; ++i) {
    g.part_offset[i] = off;
    g.part_bytes[i] = sz[i];
    off += sz[i];
  }
  g.slot_bytes = off;
  if (off % kAlign) throw std::runtime_error("expert pack: slot size is not a multiple of 4096");
  return g;
}

Geometry geometry_from(const Header& h) {
  Geometry g;
  g.n_layers = int(h.n_layers);
  g.n_mtp = int(h.n_mtp);
  g.n_experts = int(h.n_experts);
  g.w13_out = int(h.w13_out);
  g.w13_in = int(h.w13_in);
  g.w2_out = int(h.w2_out);
  g.w2_in = int(h.w2_in);
  for (int i = 0; i < kParts; ++i) {
    g.part_offset[i] = h.part_offset[i];
    g.part_bytes[i] = h.part_bytes[i];
  }
  g.slot_bytes = h.slot_bytes;
  return g;
}

void expert_tensors(const Checkpoint& ck, const Geometry& g, int group, int expert, const TensorInfo* out[kParts]) {
  const std::string p = expert_prefix(g, group, expert);
  static const char* const mat[3] = {"w1", "w3", "w2"};
  for (int m = 0; m < 3; ++m) {
    const int n = m < 2 ? g.w13_out : g.w2_out, k = m < 2 ? g.w13_in : g.w2_in;
    const TensorInfo& w = ck.at(p + mat[m] + ".weight");
    const TensorInfo& s = ck.at(p + mat[m] + ".scale");
    if (w.dtype != DType::I8 || s.dtype != DType::F8_E8M0) throw std::runtime_error(p + mat[m] + ": not MXFP4");
    if (w.nbytes != uint64_t(n) * k / 2 || s.nbytes != uint64_t(n) * k / 32)
      throw std::runtime_error(p + mat[m] + ": unexpected shape");
    out[2 * m] = &w;
    out[2 * m + 1] = &s;
  }
}

void repack_matrix(const uint8_t* w, const uint8_t* s, uint8_t* tw, uint8_t* ts, int n, int k) {
  const size_t rb = size_t(k) / 2, nb = size_t(k) / 64;  // code bytes per row, 32-byte blocks per row
  const size_t rs = size_t(k) / 32;
  const size_t n16 = size_t(n) / 16;
  for (size_t r = 0; r < size_t(n); ++r) {
    for (size_t j = 0; j < nb; ++j)
      std::memcpy(tw + ((j * n16 + r / 16) * 512 + (r % 16) * 32), w + r * rb + j * 32, 32);
    for (size_t b = 0; b < rs; ++b) ts[((b / 2) * n16 + r / 16) * 32 + (r % 16) * 2 + b % 2] = s[r * rs + b];
  }
}

void unrepack_matrix(const uint8_t* tw, const uint8_t* ts, uint8_t* w, uint8_t* s, int n, int k) {
  const size_t rb = size_t(k) / 2, nb = size_t(k) / 64;
  const size_t rs = size_t(k) / 32;
  const size_t n16 = size_t(n) / 16;
  for (size_t r = 0; r < size_t(n); ++r) {
    for (size_t j = 0; j < nb; ++j)
      std::memcpy(w + r * rb + j * 32, tw + ((j * n16 + r / 16) * 512 + (r % 16) * 32), 32);
    for (size_t b = 0; b < rs; ++b) s[r * rs + b] = ts[((b / 2) * n16 + r / 16) * 32 + (r % 16) * 2 + b % 2];
  }
}

const uint8_t* read_aligned(int fd, uint64_t off, uint64_t len, uint8_t* buf) {
  const uint64_t a0 = off / kAlign * kAlign;
  const uint64_t a1 = align_up(off + len);
  uint64_t done = 0;
  while (done < a1 - a0) {
    const ssize_t r = ::pread(fd, buf + done, size_t(a1 - a0 - done), off_t(a0 + done));
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("pread: ") + std::strerror(errno));
    }
    if (r == 0) break;  // EOF inside the aligned tail
    done += uint64_t(r);
  }
  if (done < off + len - a0) throw std::runtime_error("pread: short read");
  return buf + (off - a0);
}

uint8_t* alloc_aligned(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, kAlign, bytes)) throw std::runtime_error("posix_memalign failed");
  return static_cast<uint8_t*>(p);
}
void free_aligned(void* p) { std::free(p); }

uint64_t config_hash(const Checkpoint& ck) {
  std::ifstream f(ck.dir() + "/config.json", std::ios::binary);
  if (!f) throw std::runtime_error("cannot open config.json");
  std::ostringstream s;
  s << f.rdbuf();
  const std::string t = s.str();
  return xxh64(t.data(), t.size());
}

uint64_t source_hash(const Checkpoint& ck, const Geometry& g) {
  std::string acc;
  for (int grp = 0; grp < g.n_layers + g.n_mtp; ++grp)
    for (int e = 0; e < g.n_experts; ++e) {
      const TensorInfo* t[kParts];
      expert_tensors(ck, g, grp, e, t);
      for (int i = 0; i < kParts; ++i) {
        acc += std::filesystem::path(ck.shard_paths().at(size_t(t[i]->file))).filename().string();
        acc += ':' + std::to_string(t[i]->offset) + ':' + std::to_string(t[i]->nbytes) + ';';
      }
    }
  return xxh64(acc.data(), acc.size());
}

Header read_header(int fd) {
  uint8_t* buf = alloc_aligned(aligned_buffer_bytes(kHeaderBytes));
  Header h;
  try {
    const uint8_t* p = read_aligned(fd, 0, kHeaderBytes, buf);
    std::memcpy(&h, p, sizeof h);
  } catch (...) {
    free_aligned(buf);
    throw;
  }
  free_aligned(buf);
  if (h.magic != kMagic) throw std::runtime_error("expert pack: bad magic");
  if (h.version != kVersion) throw std::runtime_error("expert pack: unsupported version");
  if (h.header_hash != xxh64(&h, offsetof(Header, header_hash))) throw std::runtime_error("expert pack: header hash");
  return h;
}

}  // namespace unholy::model::pack

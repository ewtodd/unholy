#include "model/safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace unholy::model {

namespace {

struct DTypeName {
  DType d;
  const char* name;
  size_t size;
};
constexpr DTypeName kDTypes[] = {
    {DType::F64, "F64", 8},         {DType::F32, "F32", 4},         {DType::F16, "F16", 2},
    {DType::BF16, "BF16", 2},       {DType::F8_E4M3, "F8_E4M3", 1}, {DType::F8_E5M2, "F8_E5M2", 1},
    {DType::F8_E8M0, "F8_E8M0", 1}, {DType::I64, "I64", 8},         {DType::I32, "I32", 4},
    {DType::I16, "I16", 2},         {DType::I8, "I8", 1},           {DType::U8, "U8", 1},
    {DType::BOOL, "BOOL", 1},
};

std::string read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

DType parse_dtype(std::string_view s) {
  for (const auto& d : kDTypes)
    if (s == d.name) return d.d;
  return DType::Unknown;
}

const char* dtype_name(DType d) {
  for (const auto& e : kDTypes)
    if (e.d == d) return e.name;
  return "UNKNOWN";
}

size_t dtype_size(DType d) {
  for (const auto& e : kDTypes)
    if (e.d == d) return e.size;
  return 0;
}

std::unique_ptr<Checkpoint> Checkpoint::open(const std::string& dir, const std::vector<std::string>& extra) {
  std::unique_ptr<Checkpoint> ck(new Checkpoint());
  ck->dir_ = dir;
  const fs::path root(dir);
  if (fs::exists(root / "config.json")) ck->config_ = gufo::json::parse(read_file(root / "config.json"));
  if (fs::exists(root / "quantization_config.json"))
    ck->quant_config_ = gufo::json::parse(read_file(root / "quantization_config.json"));

  std::vector<std::string> shards;
  if (fs::exists(root / "model.safetensors.index.json")) {
    const auto index = gufo::json::parse(read_file(root / "model.safetensors.index.json"));
    const auto* wm = index.find("weight_map");
    if (!wm || !wm->is_object()) throw std::runtime_error(dir + ": index has no weight_map");
    std::set<std::string> uniq;
    for (const auto& [name, file] : wm->members()) uniq.insert(file.str());
    shards.assign(uniq.begin(), uniq.end());
  } else if (fs::exists(root / "model.safetensors")) {
    shards.push_back("model.safetensors");
  } else {
    throw std::runtime_error(dir + ": no model.safetensors(.index.json)");
  }
  for (const auto& s : shards) ck->add_shard((root / s).string());
  for (const auto& s : extra)  // side files outside the index (exllamav3's mtp.safetensors), when present
    if (fs::exists(root / s)) ck->add_shard((root / s).string());

  ck->by_name_.reserve(ck->tensors_.size());
  for (size_t i = 0; i < ck->tensors_.size(); ++i) ck->by_name_.emplace(ck->tensors_[i].name, i);
  return ck;
}

void Checkpoint::add_shard(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("open " + path + ": " + std::strerror(errno));
  struct stat st{};
  if (fstat(fd, &st) != 0) throw std::runtime_error("stat " + path);
  const size_t size = size_t(st.st_size);
  void* base = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
  if (base == MAP_FAILED) throw std::runtime_error("mmap " + path + ": " + std::strerror(errno));

  const int file_idx = int(maps_.size());
  maps_.push_back({fd, base, size});
  shard_paths_.push_back(path);

  // Header: u64 little-endian length, then that many bytes of JSON.
  uint64_t hlen = 0;
  std::memcpy(&hlen, base, 8);
  if (8 + hlen > size) throw std::runtime_error(path + ": bad safetensors header length");
  const auto header = gufo::json::parse(
      std::string_view(static_cast<const char*>(base) + 8, size_t(hlen)));
  const uint64_t data_start = 8 + hlen;

  tensors_.reserve(tensors_.size() + header.members().size());
  for (const auto& [name, v] : header.members()) {
    if (name == "__metadata__") continue;
    TensorInfo t;
    t.name = name;
    t.dtype = parse_dtype(v.member_str("dtype"));
    if (const auto* sh = v.find("shape"))
      for (const auto& d : sh->items()) t.shape.push_back(int64_t(d.as_size()));
    const auto* off = v.find("data_offsets");
    if (!off || off->size() != 2) throw std::runtime_error(path + ": " + name + " has no data_offsets");
    const uint64_t b = off->items()[0].as_size(), e = off->items()[1].as_size();
    t.file = file_idx;
    t.offset = data_start + b;
    t.nbytes = e - b;
    if (t.offset + t.nbytes > size) throw std::runtime_error(path + ": " + name + " out of bounds");
    if (t.dtype != DType::Unknown && uint64_t(t.numel()) * dtype_size(t.dtype) != t.nbytes)
      throw std::runtime_error(path + ": " + name + " size mismatch");
    tensors_.push_back(std::move(t));
  }
}

Checkpoint::~Checkpoint() {
  for (auto& m : maps_) {
    if (m.base) munmap(m.base, m.size);
    if (m.fd >= 0) ::close(m.fd);
  }
}

const TensorInfo* Checkpoint::find(std::string_view name) const {
  auto it = by_name_.find(name);
  return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

const TensorInfo& Checkpoint::at(std::string_view name) const {
  if (const auto* t = find(name)) return *t;
  throw std::runtime_error(dir_ + ": no tensor " + std::string(name));
}

std::span<const std::byte> Checkpoint::bytes(const TensorInfo& t) const {
  const auto& m = maps_.at(size_t(t.file));
  return {static_cast<const std::byte*>(m.base) + t.offset, size_t(t.nbytes)};
}

uint64_t Checkpoint::total_bytes() const {
  uint64_t n = 0;
  for (const auto& t : tensors_) n += t.nbytes;
  return n;
}

}  // namespace unholy::model

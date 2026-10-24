// A sharded safetensors checkpoint: config, tensor index, and mmap'd payloads.
//
// Both checkpoints this engine reads are plain sharded safetensors with a
// `model.safetensors.index.json`: the native DeepSeek release (fp8/MXFP4/bf16) and exllamav3's EXL3
// output (trellis/suh/svh/mul1 per linear). Format-specific views (EXL3 linears, fp8 block-scaled
// weights, MXFP4 experts) sit on top of this in their own files.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gufo/json.hpp"

namespace unholy::model {

enum class DType : uint8_t { F64, F32, F16, BF16, F8_E4M3, F8_E5M2, F8_E8M0, I64, I32, I16, I8, U8, BOOL, Unknown };

DType parse_dtype(std::string_view s);
const char* dtype_name(DType d);
size_t dtype_size(DType d);

struct TensorInfo {
  std::string name;
  DType dtype = DType::Unknown;
  std::vector<int64_t> shape;
  int file = -1;        // index into Checkpoint's shard list
  uint64_t offset = 0;  // absolute byte offset of the payload in the shard file
  uint64_t nbytes = 0;

  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

class Checkpoint {
 public:
  // `dir` holds model.safetensors.index.json (or a single model.safetensors) and config.json; `extra` names side
  // files in it whose tensors join the index when they exist (exllamav3 keeps the MTP layer in mtp.safetensors).
  static std::unique_ptr<Checkpoint> open(const std::string& dir, const std::vector<std::string>& extra = {});
  ~Checkpoint();
  Checkpoint(const Checkpoint&) = delete;
  Checkpoint& operator=(const Checkpoint&) = delete;

  const std::string& dir() const { return dir_; }
  const gufo::json::Value& config() const { return config_; }
  // Optional side files (exllamav3 writes quantization_config.json); null value if absent.
  const gufo::json::Value& quant_config() const { return quant_config_; }

  const TensorInfo* find(std::string_view name) const;
  const TensorInfo& at(std::string_view name) const;  // throws if absent
  const std::vector<TensorInfo>& tensors() const { return tensors_; }

  // Payload bytes (mmap'd; page faults on first touch).
  std::span<const std::byte> bytes(const TensorInfo& t) const;
  template <typename T>
  std::span<const T> as(const TensorInfo& t) const {
    auto b = bytes(t);
    return {reinterpret_cast<const T*>(b.data()), b.size() / sizeof(T)};
  }

  uint64_t total_bytes() const;
  const std::vector<std::string>& shard_paths() const { return shard_paths_; }

 private:
  Checkpoint() = default;
  void add_shard(const std::string& path);

  struct Map {
    int fd = -1;
    void* base = nullptr;
    size_t size = 0;
  };
  std::string dir_;
  gufo::json::Value config_;
  gufo::json::Value quant_config_;
  std::vector<std::string> shard_paths_;
  std::vector<Map> maps_;
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string_view, size_t> by_name_;
};

}  // namespace unholy::model

// Streams checkpoint tensors into device memory.
//
// mmap + hipMemcpy from pageable memory faults the file in a page at a time and, once the 168 GB
// checkpoint no longer fits the page cache, thrashes it (minutes per pass). Instead: O_DIRECT reads of
// 4 KiB-aligned ranges straight into pinned staging buffers (the NVMe DMAs into them, no page-cache
// copy), double-buffered against async host-to-device copies. Falls back to buffered pread when the
// filesystem refuses O_DIRECT.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/safetensors.hpp"

namespace unholy::model {

class Uploader {
 public:
  explicit Uploader(const Checkpoint& ck, size_t chunk_bytes = size_t(64) << 20);
  ~Uploader();
  Uploader(const Uploader&) = delete;
  Uploader& operator=(const Uploader&) = delete;

  // Copies tensor bytes [0, t.nbytes) to `dst` (device or host-accessible) using `st`. Returns
  // once the data is in a staging buffer and its copy is queued; call sync() before using dst.
  void upload(const TensorInfo& t, void* dst, hipStream_t st);
  // Many tensors at once: sorted by file offset and merged into reads of up to one chunk, so a layer's
  // ~1500 tensors cost a few dozen large reads.
  struct Item {
    const TensorInfo* t;
    void* dst;
  };
  void upload_many(std::vector<Item> items, hipStream_t st);
  void sync();

  uint64_t bytes_read() const { return bytes_; }

 private:
  const uint8_t* read_range(int file, uint64_t off, uint64_t len, uint8_t* buf);

  const Checkpoint& ck_;
  size_t chunk_;
  std::vector<int> direct_fds_;  // per shard, -1 if O_DIRECT unavailable
  std::vector<int> plain_fds_;
  uint8_t* staging_[2] = {nullptr, nullptr};
  hipEvent_t done_[2] = {nullptr, nullptr};
  bool pending_[2] = {false, false};
  int slot_ = 0;
  uint64_t bytes_ = 0;
};

}  // namespace unholy::model

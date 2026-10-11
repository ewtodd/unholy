// Bump allocator over large device slabs. Weights are tens of thousands of tensors; individual
// hipMalloc calls round each up (a 3 MiB EXL3 trellis takes 4 MiB), which wastes a quarter of VRAM.
#pragma once

#include <cstddef>
#include <vector>

#include "rt/hip.hpp"

namespace unholy::rt {

class Arena {
 public:
  explicit Arena(int dev, size_t slab_bytes = size_t(1) << 30) : dev_(dev), slab_(slab_bytes) {}
  void* alloc(size_t bytes, size_t align = 256) {
    off_ = (off_ + align - 1) / align * align;
    if (slabs_.empty() || off_ + bytes > slabs_.back().size()) {
      slabs_.emplace_back(dev_, bytes > slab_ ? bytes : slab_);
      off_ = 0;
    }
    void* p = static_cast<char*>(slabs_.back().data()) + off_;
    off_ += bytes;
    used_ += bytes;
    return p;
  }
  size_t used() const { return used_; }
  size_t reserved() const {
    size_t r = 0;
    for (const auto& s : slabs_) r += s.size();
    return r;
  }
  int device() const { return dev_; }

 private:
  int dev_;
  size_t slab_;
  size_t off_ = 0, used_ = 0;
  std::vector<DeviceBuffer> slabs_;
};

}  // namespace unholy::rt

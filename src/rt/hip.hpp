// HIP error handling and small RAII helpers shared by host code.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#define UH_CHECK(expr)                                                                         \
  do {                                                                                         \
    hipError_t uh_e_ = (expr);                                                                 \
    if (uh_e_ != hipSuccess)                                                                   \
      throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + \
                               #expr + " -> " + hipGetErrorString(uh_e_));                     \
  } while (0)

#define UH_CHECK_LAUNCH() UH_CHECK(hipGetLastError())

namespace unholy::rt {

// Sets the current device for the scope and restores the previous one.
class DeviceGuard {
 public:
  explicit DeviceGuard(int dev) {
    UH_CHECK(hipGetDevice(&prev_));
    if (dev != prev_) UH_CHECK(hipSetDevice(dev));
  }
  ~DeviceGuard() { (void)hipSetDevice(prev_); }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int prev_ = 0;
};

// Owning device allocation (move-only).
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(int dev, size_t bytes) : dev_(dev), bytes_(bytes) {
    if (bytes == 0) return;
    DeviceGuard g(dev);
    UH_CHECK(hipMalloc(&ptr_, bytes));
  }
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(DeviceBuffer&& o) noexcept { *this = std::move(o); }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      reset();
      dev_ = o.dev_;
      bytes_ = o.bytes_;
      ptr_ = o.ptr_;
      o.ptr_ = nullptr;
      o.bytes_ = 0;
    }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void reset() {
    if (ptr_) {
      DeviceGuard g(dev_);
      (void)hipFree(ptr_);
      ptr_ = nullptr;
    }
    bytes_ = 0;
  }
  void* data() const { return ptr_; }
  template <typename T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }
  size_t size() const { return bytes_; }
  int device() const { return dev_; }

 private:
  int dev_ = 0;
  size_t bytes_ = 0;
  void* ptr_ = nullptr;
};

}  // namespace unholy::rt

#include "model/loader.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#include "rt/hip.hpp"

namespace unholy::model {

namespace {
constexpr uint64_t kAlign = 4096;
}

Uploader::Uploader(const Checkpoint& ck, size_t chunk_bytes) : ck_(ck), chunk_(chunk_bytes) {
  chunk_ = (chunk_ + kAlign - 1) / kAlign * kAlign;
  for (const auto& path : ck.shard_paths()) {
    direct_fds_.push_back(::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC));
    const int p = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (p < 0) throw std::runtime_error("open " + path + ": " + std::strerror(errno));
    posix_fadvise(p, 0, 0, POSIX_FADV_SEQUENTIAL);
    plain_fds_.push_back(p);
  }
  for (int i = 0; i < 2; ++i) {
    // hipHostMalloc is page-aligned; one alignment unit of slack covers the aligned superset.
    UH_CHECK(hipHostMalloc(reinterpret_cast<void**>(&staging_[i]), chunk_ + 2 * kAlign, hipHostMallocPortable));
    UH_CHECK(hipEventCreateWithFlags(&done_[i], hipEventDisableTiming));
  }
}

Uploader::~Uploader() {
  for (int i = 0; i < 2; ++i) {
    if (pending_[i]) (void)hipEventSynchronize(done_[i]);
    if (done_[i]) (void)hipEventDestroy(done_[i]);
    if (staging_[i]) (void)hipHostFree(staging_[i]);
  }
  for (int fd : direct_fds_)
    if (fd >= 0) ::close(fd);
  for (int fd : plain_fds_) ::close(fd);
}

// Reads file bytes [off, off + len) into `buf` (page-aligned, len + 2*kAlign bytes); returns the
// pointer to byte `off` inside it.
const uint8_t* Uploader::read_range(int file, uint64_t off, uint64_t len, uint8_t* buf) {
  const int dfd = direct_fds_[size_t(file)];
  if (dfd >= 0) {
    const uint64_t a0 = off / kAlign * kAlign;
    const uint64_t a1 = (off + len + kAlign - 1) / kAlign * kAlign;
    uint64_t done = 0;
    while (done < a1 - a0) {
      const ssize_t r = ::pread(dfd, buf + done, size_t(a1 - a0 - done), off_t(a0 + done));
      if (r < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error(std::string("pread(O_DIRECT): ") + std::strerror(errno));
      }
      if (r == 0) break;  // EOF inside the aligned tail: those bytes are never used
      done += uint64_t(r);
    }
    if (done < off + len - a0) throw std::runtime_error("pread(O_DIRECT): short read");
    return buf + (off - a0);
  }
  uint64_t done = 0;
  while (done < len) {
    const ssize_t r = ::pread(plain_fds_[size_t(file)], buf + done, size_t(len - done), off_t(off + done));
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("pread: ") + std::strerror(errno));
    }
    if (r == 0) throw std::runtime_error("pread: unexpected EOF");
    done += uint64_t(r);
  }
  return buf;
}

void Uploader::upload(const TensorInfo& t, void* dst, hipStream_t st) {
  for (uint64_t pos = 0; pos < t.nbytes; pos += chunk_) {
    const uint64_t len = std::min<uint64_t>(chunk_, t.nbytes - pos);
    const int s = slot_;
    slot_ ^= 1;
    if (pending_[s]) UH_CHECK(hipEventSynchronize(done_[s]));
    const uint8_t* src = read_range(t.file, t.offset + pos, len, staging_[s]);
    UH_CHECK(hipMemcpyAsync(static_cast<uint8_t*>(dst) + pos, src, size_t(len), hipMemcpyHostToDevice, st));
    UH_CHECK(hipEventRecord(done_[s], st));
    pending_[s] = true;
    bytes_ += len;
  }
}

void Uploader::upload_many(std::vector<Item> items, hipStream_t st) {
  std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
    return a.t->file != b.t->file ? a.t->file < b.t->file : a.t->offset < b.t->offset;
  });
  constexpr uint64_t kMaxGap = uint64_t(1) << 20;  // read through small holes rather than seek
  size_t i = 0;
  while (i < items.size()) {
    const TensorInfo& first = *items[i].t;
    if (first.nbytes >= chunk_) {  // large: its own chunked stream
      upload(first, items[i].dst, st);
      ++i;
      continue;
    }
    // Group [i, j) spanning [lo, hi) within one chunk.
    const uint64_t lo = first.offset;
    uint64_t hi = first.offset + first.nbytes;
    size_t j = i + 1;
    while (j < items.size() && items[j].t->file == first.file && items[j].t->offset >= hi &&
           items[j].t->offset - hi <= kMaxGap && items[j].t->offset + items[j].t->nbytes - lo <= chunk_) {
      hi = items[j].t->offset + items[j].t->nbytes;
      ++j;
    }
    const int s = slot_;
    slot_ ^= 1;
    if (pending_[s]) UH_CHECK(hipEventSynchronize(done_[s]));
    const uint8_t* src = read_range(first.file, lo, hi - lo, staging_[s]);
    for (size_t k = i; k < j; ++k)
      UH_CHECK(hipMemcpyAsync(items[k].dst, src + (items[k].t->offset - lo), size_t(items[k].t->nbytes),
                              hipMemcpyHostToDevice, st));
    UH_CHECK(hipEventRecord(done_[s], st));
    pending_[s] = true;
    bytes_ += hi - lo;
    i = j;
  }
}

void Uploader::sync() {
  for (int i = 0; i < 2; ++i)
    if (pending_[i]) {
      UH_CHECK(hipEventSynchronize(done_[i]));
      pending_[i] = false;
    }
}

}  // namespace unholy::model

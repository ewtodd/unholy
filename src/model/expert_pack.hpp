// The expert pack: DSV4's routed experts (MXFP4) in the layout the grouped kernels read, one slot per expert, so a
// slot is one O_DIRECT read straight into the expert store (docs/PLAN-strix-only.md, "The SSDs").
//
//   [4 KiB header][zero up to 1 MiB][slot 0][slot 1]...[slot N-1][checksum table]
//
// Slot = [w1 codes | w1 scales | w3 codes | w3 scales | w2 codes | w2 scales], each part already repacked like
// kernels::mx_repack (kernels/mx_moe.hip, mx_repack_kernel). Slot order: main layers 0..n_layers-1, then the DSpark
// stages mtp.0.., each with experts 0..n_experts-1. The header stores a fingerprint of the source checkpoint.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/safetensors.hpp"

namespace unholy::model::pack {

constexpr uint64_t kMagic = 0x3150584c4f48494eull;  // "NIHOLXP1" in memory order
constexpr uint32_t kVersion = 1;
constexpr uint64_t kAlign = 4096;
constexpr uint64_t kHeaderBytes = 4096;
constexpr uint64_t kFirstSlotOffset = uint64_t(1) << 20;
constexpr int kParts = 6;  // w1 codes, w1 scales, w3 codes, w3 scales, w2 codes, w2 scales

struct Header {  // lives in the first 4 KiB, rest zero
  uint64_t magic = kMagic;
  uint32_t version = kVersion;
  uint32_t header_bytes = uint32_t(kHeaderBytes);
  uint32_t n_layers = 0, n_mtp = 0, n_experts = 0, reserved0 = 0;
  uint64_t n_slots = 0;
  uint64_t slot_bytes = 0;
  uint64_t slots_offset = 0;
  uint64_t checksum_offset = 0;  // table of n_slots x {u64 xxh64 of the slot, u64 present}
  uint64_t part_offset[kParts] = {};
  uint64_t part_bytes[kParts] = {};
  uint32_t w13_out = 0, w13_in = 0, w2_out = 0, w2_in = 0;  // w1 / w3: [out, in]; w2: [out, in]
  uint64_t config_hash = 0;  // xxh64 of config.json
  uint64_t source_hash = 0;  // xxh64 over every slot's six (shard name, offset, size)
  uint64_t reserved[8] = {};
  uint64_t header_hash = 0;  // xxh64 of the bytes before this field
};
static_assert(sizeof(Header) <= kHeaderBytes);

struct ChecksumEntry {
  uint64_t hash;
  uint64_t present;
};

uint64_t xxh64(const void* data, size_t len, uint64_t seed = 0);

inline uint64_t align_up(uint64_t x, uint64_t a = kAlign) { return (x + a - 1) / a * a; }

// Slot geometry from the checkpoint's index and the shapes of the first expert's tensors.
struct Geometry {
  int n_layers = 0, n_mtp = 0, n_experts = 0;
  int w13_out = 0, w13_in = 0, w2_out = 0, w2_in = 0;
  uint64_t part_offset[kParts] = {}, part_bytes[kParts] = {}, slot_bytes = 0;
  size_t n_slots() const { return size_t(n_layers + n_mtp) * n_experts; }
  size_t slot_index(int group, int expert) const { return size_t(group) * n_experts + expert; }
};
Geometry geometry_of(const Checkpoint& ck);
Geometry geometry_from(const Header& h);

// Tensor name prefix of slot group g (0..n_layers+n_mtp-1): "layers.G.ffn.experts.E." or "mtp.I.ffn.experts.E.".
std::string expert_prefix(const Geometry& g, int group, int expert);
// The six source tensors of one expert in part order (w1.weight, w1.scale, w3.weight, ...); throws if one is
// missing or has the wrong dtype / shape.
void expert_tensors(const Checkpoint& ck, const Geometry& g, int group, int expert, const TensorInfo* out[kParts]);

// Repack one matrix: w [n, k/2] and s [n, k/32] into the tiled layout (tw, ts), and back. Same mapping as
// mx_repack_kernel.
void repack_matrix(const uint8_t* w, const uint8_t* s, uint8_t* tw, uint8_t* ts, int n, int k);
void unrepack_matrix(const uint8_t* tw, const uint8_t* ts, uint8_t* w, uint8_t* s, int n, int k);

// Reads file bytes [off, off + len) from an O_DIRECT fd as the aligned superset. `buf` must be 4 KiB aligned with
// aligned_buffer_bytes(len) bytes; returns a pointer to byte `off` inside it.
const uint8_t* read_aligned(int fd, uint64_t off, uint64_t len, uint8_t* buf);
inline size_t aligned_buffer_bytes(uint64_t len) { return size_t(align_up(len) + 2 * kAlign); }

uint8_t* alloc_aligned(size_t bytes);  // posix_memalign(4096), throws
void free_aligned(void* p);

// Fingerprint inputs: config.json bytes, and every slot's source tensors (shard basename, offset, size).
uint64_t config_hash(const Checkpoint& ck);
uint64_t source_hash(const Checkpoint& ck, const Geometry& g);

// Reads and validates a pack header (magic, version, header hash). Throws on any mismatch.
Header read_header(int fd);

}  // namespace unholy::model::pack

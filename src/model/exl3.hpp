// EXL3 (exllamav3) linear layers: format description and a CPU reference decoder.
//
// A linear `prefix` is stored as
//   prefix.trellis  I16 [in/16, out/16, 16*K]   one tail-biting trellis bitstream per 16x16 tile
//   prefix.suh      F16 [in]                    input sign/scale
//   prefix.svh      F16 [out]                   output scale
//   prefix.mul1 | prefix.mcg   I32 []          codebook marker (absent: the 3INST codebook)
// and the dense weight (y = x W, W is [in, out]) is
//   W = diag(suh) * Hk * W_inner * Hn * diag(svh)
// with Hk / Hn block-diagonal 128-point Sylvester Hadamards scaled by 1/sqrt(128). So a forward pass
// is: x' = H(x * suh); y' = x' W_inner; y = H(y') * svh.
//
// W_inner tile (kb, nb) covers rows [16 kb, 16 kb + 16) (input) and columns [16 nb, 16 nb + 16)
// (output). Within a tile, trellis step t (0..255) decodes the 16-bit window of the MSB-first
// bitstream ending at bit (t + 1) * K (circularly), and lands at row-major tile position
// kTensorCorePerm[t]. The bitstream is the tile's 16 * K uint16 read as little-endian uint32 words,
// each word MSB-first (exllamav3 packs with SWAP16).
//
// Everything here is the reference the GPU kernels are tested against; none of it is fast.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "model/safetensors.hpp"

namespace unholy::model {

enum class Exl3Codebook : uint8_t { ThreeInst = 0, Mcg = 1, Mul1 = 2 };

struct Exl3Linear {
  std::string prefix;
  const TensorInfo* trellis = nullptr;
  const TensorInfo* suh = nullptr;
  const TensorInfo* svh = nullptr;
  int in_features = 0;
  int out_features = 0;
  int K = 0;  // bits per weight (integer rates only; half-integer rates are not supported yet)
  Exl3Codebook cb = Exl3Codebook::ThreeInst;
};

// Tile position (row-major within 16x16) of trellis step t.
const std::array<uint16_t, 256>& exl3_tensor_core_perm();

// One trellis state (16-bit window) -> fp16 bits.
uint16_t exl3_decode_state(Exl3Codebook cb, uint32_t state);

// 16-bit state for step t of a tile bitstream (`words` = the tile as 8*K uint32).
uint32_t exl3_tile_state(const uint32_t* words, int K, int t);

std::optional<Exl3Linear> find_exl3_linear(const Checkpoint& ck, const std::string& prefix);

// W_inner as fp16 bits, row-major [in][out].
void exl3_reconstruct_inner_cpu(const Checkpoint& ck, const Exl3Linear& l, uint16_t* w_inner);

// Dense W [in][out] in float: Hadamards and scales applied in double, no intermediate rounding.
void exl3_dense_cpu(const Checkpoint& ck, const Exl3Linear& l, float* w);

// In-place 128-point Sylvester FWHT scaled by 1/sqrt(128) over x[0], x[stride], ..., x[127 * stride].
void fwht128(double* x, int stride);

}  // namespace unholy::model

// Hy4 flash's vision tower (transformers HYV4VL vision model + its "perceive" projector, PR #50903; spec in
// DECISIONS 2026-11-20): one image's 16-pixel patches -> the rows of its image pad tokens.
//
// Per image (grid gh x gw patches, raster order): the patch embedding (a linear over 3 x 16 x 16, with bias) plus a
// learned 128 x 128 position table resampled bilinearly to the grid; 27 pre-LN blocks (LayerNorm with bias,
// attention over the whole image with 16 heads of 72 and biases, no rope; LayerNorm, a GELU (erf) MLP of 4304);
// no final norm; the projector: RMSNorm, a 2 x 2 stride-2 conv to 4608, GELU, a 1 x 1 conv to the text width, and a
// learned newline row after each merged row: (gh / 2) * (gw / 2 + 1) rows.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

#include "gufo/json.hpp"
#include "model/dsv4.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"

namespace unholy::model {

struct Hy4VisionConfig {
  int depth = 0, dim = 0, heads = 0, head_dim = 0, inter = 0, out_dim = 0, patch = 0, merge = 0, edge = 0;
  int max_patches = 0;  // max_vit_seq_len
  float eps = 0;
  // From config.json's vision_config; depth 0 when there is none.
  static Hy4VisionConfig from_hf(const gufo::json::Value& top);
};

struct Hy4VisionWeights {
  struct Block {
    DevTensor ln1_w, ln1_b, o_w, o_b, ln2_w, ln2_b, fc2_b;
    rt::DeviceBuffer qkv_w, qkv_b;  // q, k, v rows per head padded to 80: [3][heads][80][dim], [3][heads][80]
    rt::DeviceBuffer fc1_w, fc1_b;  // [inter_pad][dim], [inter_pad] (zero rows past inter)
    rt::DeviceBuffer fc2_w;         // [dim][inter_pad] (zero columns past inter)
  };
  int inter_pad = 0;
  DevTensor patch_w, patch_b, pos, norm, conv1_w, conv1_b, conv2_w, conv2_b, newline;
  std::vector<Block> blocks;
};
Hy4VisionWeights load_hy4_vision(const Checkpoint& ck, const Hy4VisionConfig& c, int dev, hipStream_t st);
size_t hy4_vision_weight_bytes(const Checkpoint& ck);

struct Hy4VisionScratch {
  int max_patches = 0;
  rt::DeviceBuffer patches, x, h, o, qkv, acc, mlp, unfold, conv, merged, src;
};
// Buffers for images of up to max_patches patches.
Hy4VisionScratch make_hy4_vision_scratch(const Hy4VisionConfig& c, int max_patches, int dev);
size_t hy4_vision_scratch_bytes(const Hy4VisionConfig& c, int max_patches);

// patches: host [gh * gw][3 * patch * patch] bf16, raster order. out: device [(gh / 2) * (gw / 2 + 1)][out_dim] bf16.
void hy4_vision_encode(const Hy4VisionConfig& c, const Hy4VisionWeights& w, Hy4VisionScratch& s, int gh, int gw,
                       const std::vector<uint16_t>& patches, uint16_t* out, hipStream_t st);

}  // namespace unholy::model

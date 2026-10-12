// The vision tower's forward (inference/vision.py ViT + Aligner, model.py merge_image_embeddings): one
// image's patches -> the embedding rows of its whole token block (learned rows for start / pad /
// newline / end, the aligner's outputs in the N-layout order for the image slots).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "model/dsv4.hpp"
#include "rt/hip.hpp"
#include "vision/image.hpp"

namespace unholy::model {

struct VisionScratch {
  int max_patches = 0;
  rt::DeviceBuffer patches, x, h, qkv, o, f32, mlp, unfold, gelu, aligned, cos, sin, src;
};
// Buffers for images of up to the config's token budget (max_tokens * downsample^2 patches).
VisionScratch make_vision_scratch(const Dsv4Config& c, int dev);

// out: device [block.types.size()][dim] bf16, the block's embedding rows in sequence order.
void vision_encode(const Dsv4Config& c, const VisionWeights& w, VisionScratch& s, const vision::Patches& p,
                   const vision::ImageBlock& block, uint16_t* out, hipStream_t st);

}  // namespace unholy::model

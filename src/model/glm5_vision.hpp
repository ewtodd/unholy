// GLM-5.3-Flash's vision tower (transformers Glm5NextVisionModel + its patch merger): one image's patches -> the
// rows of its image tokens. The weights are the checkpoint's vision.* tensors, the release's bf16 tower beside an
// EXL3 quant (tools/box/extract-glm-vision.py).
//
// Per image: a Conv3d patch embedding (a linear over 3 x 2 x 14 x 14), 24 blocks of RMSNorm, attention over the
// whole image (qkv with bias, per-head q / k RMSNorm, axial 2-D rope over the full head: the first 16 frequencies
// by row, the next 16 by column), RMSNorm and a SwiGLU with biases and inputs clamped to 10; a final RMSNorm; a
// 2 x 2 stride-2 Conv2d over each merge group (the processor emits a group's 4 patches together, so it is a
// linear over 4 consecutive rows); the merger: a linear, LayerNorm, GELU, and a clamped SwiGLU to the text width.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

#include "gufo/json.hpp"
#include "model/dsv4.hpp"
#include "model/safetensors.hpp"
#include "rt/hip.hpp"
#include "vision/image.hpp"

namespace unholy::model {

struct Glm5VisionConfig {
  int depth = 0, dim = 0, heads = 0, inter = 0, out_dim = 0, proj_inter = 0;
  int in_channels = 0, patch = 0, temporal = 0, merge = 0;
  float eps = 0, swiglu_limit = 0, rope_theta = 0;
  // From config.json's vision_config; depth 0 when there is none.
  static Glm5VisionConfig from_hf(const gufo::json::Value& top);
  vision::GlmVisionConfig preprocess(int max_tokens) const;
};

struct Glm5VisionWeights {
  struct Block {
    DevTensor norm1, qkv_w, qkv_b, q_norm, k_norm, proj_w, proj_b, norm2, down_w, down_b;
    rt::DeviceBuffer gate_up_w, gate_up_b;  // gate_proj rows then up_proj rows: [2 inter][dim], [2 inter]
  };
  rt::DeviceBuffer patch_w;  // [dim][kpad]: the Conv3d with K zero-padded to a multiple of 32 (the GEMM's step)
  int kpad = 0;
  DevTensor patch_b, post_norm, ds_b, m_proj, m_norm_w, m_norm_b, m_down;
  rt::DeviceBuffer ds_w;       // downsample [out_dim][(i * merge + j) * dim + c], from Conv2d [out_dim][dim][i][j]
  rt::DeviceBuffer m_gate_up;  // merger gate_proj rows then up_proj rows: [2 proj_inter][out_dim]
  std::vector<Block> blocks;
};
Glm5VisionWeights load_glm5_vision(const Checkpoint& ck, const Glm5VisionConfig& c, int dev, hipStream_t st);

struct Glm5VisionScratch {
  int max_patches = 0;
  rt::DeviceBuffer patches, x, h, o, qkv, acc, mlp, cos, sin;
};
// Buffers for images of up to max_tokens tokens (max_tokens * merge^2 patches); row-wise layers run in chunks of
// rows, so the fp32 accumulator stays a fixed size.
Glm5VisionScratch make_glm5_vision_scratch(const Glm5VisionConfig& c, int max_tokens, int dev);
size_t glm5_vision_scratch_bytes(const Glm5VisionConfig& c, int max_tokens);

// out: device [grid_h * grid_w / merge^2][out_dim] bf16, the image tokens' rows in sequence order.
void glm5_vision_encode(const Glm5VisionConfig& c, const Glm5VisionWeights& w, Glm5VisionScratch& s,
                        const vision::GlmPatches& p, uint16_t* out, hipStream_t st);

}  // namespace unholy::model

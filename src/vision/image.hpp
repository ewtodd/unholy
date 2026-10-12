// Image preprocessing for DSV4's vision encoder, bit-exact with the reference (inference/
// image_processor.py: PIL decode + convert("RGB"), resize / ImageOps.pad with PIL's bicubic
// resampling, [-1, 1] normalisation to bf16, 14-pixel patches) and its token layout
// (build_image_block: the N-layout grid that puts each 2 x 2 patch square into one 4-token
// compression block).
#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace unholy::vision {

struct VisionConfig {
  int patch = 14, downsample = 3, max_tokens = 384, min_pixels = 147456, max_wh_ratio = 8;
};

struct ImageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// 8-bit RGB, rows top to bottom.
struct RgbImage {
  int w = 0, h = 0;
  std::vector<uint8_t> px;  // [h][w][3]
};

// JPEG (libjpeg-turbo, the decoder settings PIL uses), PNG (16-bit channels keep their high byte),
// WebP and GIF (first frame) -> RGB the way PIL's convert("RGB") does it (alpha dropped, palette and
// grey expanded). Throws ImageError on anything else.
RgbImage decode_image(std::span<const uint8_t> bytes);

// PIL Image.resize(size) for an RGB image: the default BICUBIC filter (a = -0.5), support scaled by
// the reduction factor, coefficients in 22-bit fixed point, horizontal pass then vertical.
RgbImage resize_bicubic(const RgbImage& in, int w, int h);

// The encoder's input for one image: patches [n_vit_h * n_vit_w][3][patch][patch] bf16 (as uint16) and
// the grid sizes (load_image).
struct Patches {
  int n_vit_h = 0, n_vit_w = 0, n_llm_h = 0, n_llm_w = 0;
  std::vector<uint16_t> data;
};
Patches preprocess(const RgbImage& img, const VisionConfig& cfg);

// Token types of an image block (build_image_block); the block's token ids are vocab + type.
enum ImageToken { kImageStart = 0, kImagePad = 1, kImage = 2, kImageNewLine = 3, kImageEnd = 4 };
struct ImageBlock {
  std::vector<int32_t> types;  // the block in sequence order, starting with its alignment padding
  std::vector<int32_t> perm;   // for each kImage slot in order: the aligner row (row-major grid index)
};
// `start_pos`: the sequence position where the block begins (it pads itself to a 4-token boundary).
ImageBlock build_image_block(int n_llm_h, int n_llm_w, int start_pos);

// One image of a request: its block's place in the token sequence, the encoder input and a hash of
// the image bytes (prefix caching tells different images with the same layout apart by it).
struct ImageInput {
  int start = 0;  // position of the block's first token (its alignment padding)
  Patches patches;
  ImageBlock block;
  uint64_t hash = 0;
};
// prepare_vl_inputs: replaces each `placeholder` token of `ids` with the block of the next image (ids
// vocab + type), decoding and preprocessing the images. Throws ImageError when the counts differ.
std::vector<ImageInput> expand_images(std::vector<int32_t>& ids, int32_t placeholder, int32_t vocab,
                                      const std::vector<std::string>& urls, const VisionConfig& cfg);

// Image bytes from an OpenAI-style URL: "data:<mime>;base64,<payload>" only (http(s) and paths are
// refused). Throws ImageError.
std::vector<uint8_t> image_bytes_from_url(const std::string& url);

}  // namespace unholy::vision

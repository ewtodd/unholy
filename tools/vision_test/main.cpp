// unholy-vision-test: image preprocessing as the server does it, for tools/ref/vision_compare.py.
//
//   unholy-vision-test IMAGE START_POS OUT
//
// Decodes IMAGE, preprocesses it (src/vision/image.hpp) and lays out its token block at START_POS;
// writes the patches (bf16 [n_vit_h * n_vit_w][3][14][14]) to OUT and prints one JSON line:
// {"n_vit_h", "n_vit_w", "n_llm_h", "n_llm_w", "w", "h", "types": [...], "perm": [...]}.
//
//   unholy-vision-test glm IMAGE OUT [MAX_TOKENS]
//
// GLM-5.3-Flash's preprocessing (vision::glm_preprocess), for tools/ref/glm_vision_compare.py: the patches (bf16
// [grid_h * grid_w][3 * 2 * 14 * 14]) to OUT and {"grid_h", "grid_w", "tokens", "w", "h"}.
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "text/json.hpp"
#include "vision/image.hpp"

using namespace unholy;

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "expand") {
    // expand IN.json: {"ids", "urls", "placeholder", "vocab"} -> {"ids", "images": [{start, n_vit_h, n_vit_w,
    // n_llm_h, n_llm_w, perm}]}
    try {
      std::ifstream f(argv[2]);
      const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      const auto in = json::parse(text);
      std::vector<int32_t> ids;
      for (const auto& v : in.at("ids").items()) ids.push_back(int32_t(v.as_int()));
      std::vector<std::string> urls;
      for (const auto& v : in.at("urls").items()) urls.push_back(v.str());
      const auto imgs = vision::expand_images(ids, int32_t(in.at("placeholder").as_int()), int32_t(in.at("vocab").as_int()),
                                              urls, vision::VisionConfig{});
      std::printf("{\"ids\": [");
      for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
      std::printf("], \"images\": [");
      for (size_t k = 0; k < imgs.size(); ++k) {
        const auto& m = imgs[k];
        std::printf("%s{\"start\": %d, \"n_vit_h\": %d, \"n_vit_w\": %d, \"n_llm_h\": %d, \"n_llm_w\": %d, \"perm\": [", k ? ", " : "",
                    m.start, m.patches.n_vit_h, m.patches.n_vit_w, m.patches.n_llm_h, m.patches.n_llm_w);
        for (size_t i = 0; i < m.block.perm.size(); ++i) std::printf("%s%d", i ? ", " : "", m.block.perm[i]);
        std::printf("]}");
      }
      std::printf("]}\n");
    } catch (const std::exception& e) {
      std::printf("{\"error\": \"%s\"}\n", e.what());
      return 1;
    }
    return 0;
  }
  if ((argc == 4 || argc == 5) && std::string(argv[1]) == "glm") {
    try {
      std::ifstream f(argv[2], std::ios::binary);
      if (!f) throw std::runtime_error(std::string("cannot open ") + argv[2]);
      const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      const auto img = vision::decode_image(bytes);
      vision::GlmVisionConfig cfg;
      if (argc == 5) cfg.max_tokens = std::stoi(argv[4]);
      const auto p = vision::glm_preprocess(img, cfg);
      std::ofstream o(argv[3], std::ios::binary);
      o.write(reinterpret_cast<const char*>(p.data.data()), std::streamsize(p.data.size() * 2));
      std::printf("{\"grid_h\": %d, \"grid_w\": %d, \"tokens\": %d, \"w\": %d, \"h\": %d}\n", p.grid_h, p.grid_w,
                  p.grid_h * p.grid_w / 4, img.w, img.h);
    } catch (const std::exception& e) {
      std::printf("{\"error\": \"%s\"}\n", e.what());
      return 1;
    }
    return 0;
  }
  if (argc != 4) {
    std::fprintf(stderr, "usage: unholy-vision-test IMAGE START_POS OUT | glm IMAGE OUT [MAX_TOKENS]\n");
    return 2;
  }
  try {
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + argv[1]);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto img = vision::decode_image(bytes);
    const auto p = vision::preprocess(img, vision::VisionConfig{});
    const auto b = vision::build_image_block(p.n_llm_h, p.n_llm_w, std::stoi(argv[2]));
    std::ofstream o(argv[3], std::ios::binary);
    o.write(reinterpret_cast<const char*>(p.data.data()), std::streamsize(p.data.size() * 2));
    std::printf("{\"n_vit_h\": %d, \"n_vit_w\": %d, \"n_llm_h\": %d, \"n_llm_w\": %d, \"w\": %d, \"h\": %d, \"types\": [",
                p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w, img.w, img.h);
    for (size_t i = 0; i < b.types.size(); ++i) std::printf("%s%d", i ? ", " : "", b.types[i]);
    std::printf("], \"perm\": [");
    for (size_t i = 0; i < b.perm.size(); ++i) std::printf("%s%d", i ? ", " : "", b.perm[i]);
    std::printf("]}\n");
  } catch (const std::exception& e) {
    std::printf("{\"error\": \"%s\"}\n", e.what());
    return 1;
  }
  return 0;
}

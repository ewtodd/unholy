// unholy-hy4-preprocess IMAGE OUT_PREFIX: Hy4-flash image preprocessing (vision::hy4_preprocess), for
// tools/ref/hy4_preprocess_check.py. Writes OUT_PREFIX.rgb (the resized uint8 image, [h][w][3]) and OUT_PREFIX.bf16
// (the patches, [grid_h * grid_w][768] bf16) and prints {"grid_h", "grid_w", "tokens", "rows", "cols", "w", "h"}.
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vision/image.hpp"

using namespace unholy;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: unholy-hy4-preprocess IMAGE OUT_PREFIX\n");
    return 2;
  }
  try {
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + argv[1]);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto img = vision::decode_image(bytes);
    const vision::Hy4VisionConfig cfg;
    const auto resized = vision::hy4_resize(img, cfg);
    const auto p = vision::hy4_preprocess(img, cfg);
    std::ofstream(std::string(argv[2]) + ".rgb", std::ios::binary)
        .write(reinterpret_cast<const char*>(resized.px.data()), std::streamsize(resized.px.size()));
    std::ofstream(std::string(argv[2]) + ".bf16", std::ios::binary)
        .write(reinterpret_cast<const char*>(p.data.data()), std::streamsize(p.data.size() * 2));
    const int rows = p.grid_h / cfg.merge, cols = p.grid_w / cfg.merge + 1;
    std::printf("{\"grid_h\": %d, \"grid_w\": %d, \"tokens\": %d, \"rows\": %d, \"cols\": %d, \"w\": %d, \"h\": %d}\n",
                p.grid_h, p.grid_w, rows * cols, rows, cols, resized.w, resized.h);
  } catch (const std::exception& e) {
    std::printf("{\"error\": \"%s\"}\n", e.what());
    return 1;
  }
  return 0;
}

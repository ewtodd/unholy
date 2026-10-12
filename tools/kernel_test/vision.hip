// unholy-vit-test: the vision tower (model/vision) on one image against the reference rows from
// tools/ref/oracle/vision_ref.py (same image, block at start position 0).
//
//   unholy-vit-test --model DIR --image FILE --ref FILE [--dev N] [--reps 3]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "model/dsv4.hpp"
#include "model/safetensors.hpp"
#include "model/vision.hpp"
#include "rt/hip.hpp"
#include "vision/image.hpp"

using namespace unholy;

int main(int argc, char** argv) {
  std::string model_dir = "/scratch/dsv4-ve", image, ref, out_path;
  int dev = 0, reps = 3;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--model") model_dir = v;
    else if (a == "--image") image = v;
    else if (a == "--ref") ref = v;
    else if (a == "--dev") dev = std::stoi(v);
    else if (a == "--reps") reps = std::stoi(v);
    else if (a == "--out") out_path = v;
  }
  try {
    auto ck = model::Checkpoint::open(model_dir);
    const auto c = model::Dsv4Config::from_hf(ck->config());
    UH_CHECK(hipSetDevice(dev));
    hipStream_t st;
    UH_CHECK(hipStreamCreate(&st));
    const auto t0 = std::chrono::steady_clock::now();
    const auto w = model::load_vision(*ck, c, dev, st);
    auto s = model::make_vision_scratch(c, dev);
    std::printf("loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::ifstream f(image, std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    vision::VisionConfig vc{c.vision_patch, c.vision_downsample, c.vision_max_tokens, c.vision_min_pixels,
                            c.vision_max_wh_ratio};
    const auto p = vision::preprocess(vision::decode_image(bytes), vc);
    const auto blk = vision::build_image_block(p.n_llm_h, p.n_llm_w, 0);
    const size_t n = blk.types.size();
    rt::DeviceBuffer out(dev, n * c.dim * 2);
    double ms = 0;
    for (int r = 0; r < reps; ++r) {
      const auto a = std::chrono::steady_clock::now();
      model::vision_encode(c, w, s, p, blk, out.as<uint16_t>(), st);
      UH_CHECK(hipStreamSynchronize(st));
      ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    }
    std::vector<uint16_t> got(n * c.dim);
    UH_CHECK(hipMemcpy(got.data(), out.data(), got.size() * 2, hipMemcpyDeviceToHost));
    if (!out_path.empty()) {
      std::ofstream of(out_path, std::ios::binary);
      of.write(reinterpret_cast<const char*>(got.data()), std::streamsize(got.size() * 2));
    }
    std::printf("grid %dx%d -> %dx%d, block %zu rows, encode %.1f ms\n", p.n_vit_h, p.n_vit_w, p.n_llm_h, p.n_llm_w, n, ms);
    if (!ref.empty()) {
      std::ifstream rf(ref, std::ios::binary);
      std::vector<uint16_t> want(got.size());
      rf.read(reinterpret_cast<char*>(want.data()), std::streamsize(want.size() * 2));
      if (size_t(rf.gcount()) != want.size() * 2) throw std::runtime_error("reference has a different size");
      const auto f32 = [](uint16_t b) {
        const uint32_t u = uint32_t(b) << 16;
        float x;
        std::memcpy(&x, &u, 4);
        return x;
      };
      double num = 0, den = 0, worst_row = 0;
      size_t same = 0;
      for (size_t r = 0; r < n; ++r) {
        double rn = 0, rd = 0;
        for (int d = 0; d < c.dim; ++d) {
          const double a = f32(got[r * c.dim + size_t(d)]), b = f32(want[r * c.dim + size_t(d)]);
          rn += (a - b) * (a - b);
          rd += b * b;
          same += got[r * c.dim + size_t(d)] == want[r * c.dim + size_t(d)];
        }
        num += rn;
        den += rd;
        worst_row = std::max(worst_row, std::sqrt(rn / std::max(rd, 1e-30)));
      }
      std::printf("vs reference: rel L2 %.2e (worst row %.2e), %.1f%% of values bit-identical\n", std::sqrt(num / den),
                  worst_row, 100.0 * same / got.size());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}

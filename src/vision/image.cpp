#include "vision/image.hpp"

#include <png.h>
#include <turbojpeg.h>
#include <webp/decode.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>

namespace unholy::vision {

namespace {

// ---- decoding ----------------------------------------------------------------------------------------

RgbImage decode_jpeg(std::span<const uint8_t> b) {
  tjhandle h = tjInitDecompress();
  if (!h) throw ImageError("jpeg: cannot initialise the decoder");
  std::unique_ptr<void, int (*)(tjhandle)> guard(h, tjDestroy);
  int w = 0, ht = 0, subsamp = 0, cs = 0;
  if (tjDecompressHeader3(h, b.data(), static_cast<unsigned long>(b.size()), &w, &ht, &subsamp, &cs) != 0)
    throw ImageError(std::string("jpeg: ") + tjGetErrorStr2(h));
  RgbImage img{w, ht, {}};
  img.px.resize(size_t(w) * ht * 3);
  if (cs == TJCS_CMYK || cs == TJCS_YCCK) {
    // PIL: Adobe CMYK is stored inverted ("CMYK;I"), then convert("RGB") (Convert.c cmyk2rgb).
    std::vector<uint8_t> cmyk(size_t(w) * ht * 4);
    if (tjDecompress2(h, b.data(), static_cast<unsigned long>(b.size()), cmyk.data(), w, 0, ht, TJPF_CMYK, 0) != 0)
      throw ImageError(std::string("jpeg: ") + tjGetErrorStr2(h));
    const auto muldiv255 = [](int a, int c) {
      const int t = a * c + 128;
      return ((t >> 8) + t) >> 8;
    };
    for (size_t i = 0; i < size_t(w) * ht; ++i) {
      const int k = 255 - cmyk[i * 4 + 3], nk = 255 - k;  // inverted storage: k = 255 - stored
      for (int c = 0; c < 3; ++c) {
        const int v = 255 - cmyk[i * 4 + size_t(c)];
        img.px[i * 3 + size_t(c)] = uint8_t(std::clamp(nk - muldiv255(v, nk), 0, 255));
      }
    }
    return img;
  }
  // libjpeg's defaults, as PIL: islow IDCT, fancy upsampling, libjpeg's YCbCr -> RGB (grey replicated).
  if (tjDecompress2(h, b.data(), static_cast<unsigned long>(b.size()), img.px.data(), w, 0, ht, TJPF_RGB, 0) != 0)
    throw ImageError(std::string("jpeg: ") + tjGetErrorStr2(h));
  return img;
}

struct PngRead {
  std::span<const uint8_t> b;
  size_t off = 0;
};

RgbImage decode_png(std::span<const uint8_t> b) {
  png_structp p = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!p) throw ImageError("png: cannot initialise the decoder");
  png_infop info = png_create_info_struct(p);
  RgbImage img;
  PngRead src{b, 0};
  std::vector<uint8_t> rows;
  bool grey16 = false;
  if (setjmp(png_jmpbuf(p))) {
    png_destroy_read_struct(&p, &info, nullptr);
    throw ImageError("png: corrupt image");
  }
  png_set_read_fn(p, &src, [](png_structp ps, png_bytep out, png_size_t n) {
    auto* s = static_cast<PngRead*>(png_get_io_ptr(ps));
    if (s->off + n > s->b.size()) png_error(ps, "truncated");
    std::memcpy(out, s->b.data() + s->off, n);
    s->off += n;
  });
  png_read_info(p, info);
  const int ct = png_get_color_type(p, info), depth = png_get_bit_depth(p, info);
  img.w = int(png_get_image_width(p, info));
  img.h = int(png_get_image_height(p, info));
  // PIL modes: 16-bit grey opens as I;16 and convert("RGB") clips it to 255 (kept, quirk and all); other
  // 16-bit images keep the high byte; palette and low-depth grey expand; alpha is dropped.
  grey16 = depth == 16 && ct == PNG_COLOR_TYPE_GRAY;
  if (grey16) png_set_swap(p);  // native little-endian 16-bit for the clip below
  else if (depth == 16) png_set_strip_16(p);
  if (ct == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(p);
  if (ct == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(p);
  if (ct & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(p);
  png_set_interlace_handling(p);
  png_read_update_info(p, info);
  const size_t stride = png_get_rowbytes(p, info);
  const int ch = png_get_channels(p, info);
  rows.resize(stride * size_t(img.h));
  std::vector<png_bytep> ptrs(size_t(img.h));
  for (int y = 0; y < img.h; ++y) ptrs[size_t(y)] = rows.data() + size_t(y) * stride;
  png_read_image(p, ptrs.data());
  png_read_end(p, nullptr);
  png_destroy_read_struct(&p, &info, nullptr);
  img.px.resize(size_t(img.w) * img.h * 3);
  for (int y = 0; y < img.h; ++y)
    for (int x = 0; x < img.w; ++x) {
      uint8_t* o = &img.px[(size_t(y) * img.w + size_t(x)) * 3];
      const uint8_t* r = rows.data() + size_t(y) * stride;
      if (grey16) {
        uint16_t v;
        std::memcpy(&v, r + size_t(x) * 2, 2);
        o[0] = o[1] = o[2] = uint8_t(std::min<int>(v, 255));
      } else if (ch == 1) {
        o[0] = o[1] = o[2] = r[x];
      } else {
        std::memcpy(o, r + size_t(x) * size_t(ch), 3);
      }
    }
  return img;
}

RgbImage decode_webp(std::span<const uint8_t> b) {
  int w = 0, h = 0;
  uint8_t* rgba = WebPDecodeRGBA(b.data(), b.size(), &w, &h);
  if (!rgba) throw ImageError("webp: cannot decode");
  RgbImage img{w, h, std::vector<uint8_t>(size_t(w) * h * 3)};
  for (size_t i = 0; i < size_t(w) * h; ++i) std::memcpy(&img.px[i * 3], rgba + i * 4, 3);
  WebPFree(rgba);
  return img;
}

// ---- PIL resampling (libImaging/Resample.c) ----------------------------------------------------------

constexpr int kPrecisionBits = 32 - 8 - 2;

double bicubic(double x) {
  constexpr double a = -0.5;
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

struct Coeffs {
  int ksize = 0;
  std::vector<int> bounds;  // [out][2]: xmin, count
  std::vector<int32_t> k;   // [out][ksize], fixed point
};

Coeffs precompute(int in_size, double in0, double in1, int out_size) {
  double filterscale, scale;
  filterscale = scale = (in1 - in0) / out_size;
  if (filterscale < 1.0) filterscale = 1.0;
  const double support = 2.0 * filterscale;
  Coeffs c;
  c.ksize = int(std::ceil(support)) * 2 + 1;
  c.bounds.resize(size_t(out_size) * 2);
  std::vector<double> kk(size_t(out_size) * c.ksize, 0.0);
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0;
    const double ss = 1.0 / filterscale;
    int xmin = int(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = int(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* k = &kk[size_t(xx) * c.ksize];
    for (int x = 0; x < xmax; ++x) {
      const double w = bicubic((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x)
      if (ww != 0.0) k[x] /= ww;
    c.bounds[size_t(xx) * 2] = xmin;
    c.bounds[size_t(xx) * 2 + 1] = xmax;
  }
  c.k.resize(kk.size());
  for (size_t i = 0; i < kk.size(); ++i)
    c.k[i] = kk[i] < 0 ? int32_t(-0.5 + kk[i] * (1 << kPrecisionBits)) : int32_t(0.5 + kk[i] * (1 << kPrecisionBits));
  return c;
}

uint8_t clip8(int32_t v) { return uint8_t(std::clamp(v >> kPrecisionBits, 0, 255)); }

}  // namespace

RgbImage decode_image(std::span<const uint8_t> b) {
  if (b.size() >= 3 && b[0] == 0xff && b[1] == 0xd8 && b[2] == 0xff) return decode_jpeg(b);
  if (b.size() >= 8 && std::memcmp(b.data(), "\x89PNG\r\n\x1a\n", 8) == 0) return decode_png(b);
  if (b.size() >= 12 && std::memcmp(b.data(), "RIFF", 4) == 0 && std::memcmp(b.data() + 8, "WEBP", 4) == 0)
    return decode_webp(b);
  throw ImageError("unsupported image format (JPEG, PNG and WebP are)");
}

RgbImage resize_bicubic(const RgbImage& in, int w, int h) {
  if (w == in.w && h == in.h) return in;
  const Coeffs ch = precompute(in.w, 0.0, double(in.w), w), cv = precompute(in.h, 0.0, double(in.h), h);
  const bool need_h = w != in.w, need_v = h != in.h;
  // Horizontal pass over the rows the vertical pass reads, then the vertical pass.
  const int y_first = cv.bounds[0], y_last = cv.bounds[size_t(h) * 2 - 2] + cv.bounds[size_t(h) * 2 - 1];
  RgbImage tmp;
  const RgbImage* src = &in;
  int yoff = 0;
  if (need_h) {
    tmp.w = w;
    tmp.h = y_last - y_first;
    tmp.px.resize(size_t(tmp.w) * tmp.h * 3);
    for (int y = 0; y < tmp.h; ++y)
      for (int x = 0; x < w; ++x) {
        const int xmin = ch.bounds[size_t(x) * 2], n = ch.bounds[size_t(x) * 2 + 1];
        const int32_t* k = &ch.k[size_t(x) * ch.ksize];
        int32_t s[3] = {1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1)};
        const uint8_t* row = &in.px[(size_t(y + y_first) * in.w) * 3];
        for (int i = 0; i < n; ++i)
          for (int c = 0; c < 3; ++c) s[c] += int32_t(row[size_t(xmin + i) * 3 + size_t(c)]) * k[i];
        for (int c = 0; c < 3; ++c) tmp.px[(size_t(y) * w + size_t(x)) * 3 + size_t(c)] = clip8(s[c]);
      }
    src = &tmp;
    yoff = y_first;
  }
  if (!need_v) return need_h ? tmp : in;
  RgbImage out{src->w, h, std::vector<uint8_t>(size_t(src->w) * h * 3)};
  for (int y = 0; y < h; ++y) {
    const int ymin = cv.bounds[size_t(y) * 2] - yoff, n = cv.bounds[size_t(y) * 2 + 1];
    const int32_t* k = &cv.k[size_t(y) * cv.ksize];
    for (int x = 0; x < src->w; ++x) {
      int32_t s[3] = {1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1)};
      for (int i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c)
          s[c] += int32_t(src->px[(size_t(ymin + i) * src->w + size_t(x)) * 3 + size_t(c)]) * k[i];
      for (int c = 0; c < 3; ++c) out.px[(size_t(y) * out.w + size_t(x)) * 3 + size_t(c)] = clip8(s[c]);
    }
  }
  return out;
}

namespace {

struct Grid {
  int n_llm_h, n_llm_w, num_tokens;
};

Grid grid_tokens(int best_h, int best_w, int p, int r) {
  const int n_llm_h = (best_h / p + r - 1) / r, n_llm_w = (best_w / p + r - 1) / r;
  int num = n_llm_h * (n_llm_w + 1) + 2;
  if (n_llm_h % 2 == 1) num += n_llm_w + 1;
  num += (n_llm_h + 1) / 2 * (n_llm_w + 1) % 2 * 2;
  return {n_llm_h, n_llm_w, num};
}

struct Fit {
  int n_llm_h, n_llm_w, best_h, best_w, num_tokens;
};

int floor_div(int a, int b) { return int(std::floor(double(a) / double(b))); }

Fit solve_resize_ratio(int height, int width, int p, int r, int max_n) {
  const double ratio = double(height) / width;
  const double max_w_f = std::sqrt((max_n - 2) / ratio + 0.25) - 0.5;
  const double max_h_f = max_w_f * ratio;
  int best_w, best_h;
  if (max_w_f < 1.0) {
    const int max_w = 1;
    int max_h = floor_div(max_n - 2, max_w + 1);
    if (max_h % 2 == 1) max_h -= 1;
    best_w = max_w * p * r;
    best_h = max_h * p * r;
  } else if (max_h_f < 2.0) {
    const int max_h = 2;
    const int max_w = floor_div(max_n - 2, max_h) - 1;
    if (max_w <= 1) throw ImageError("image: no grid fits the token budget");
    best_w = max_w * p * r;
    best_h = max_h * p * r;
  } else {
    const int max_w = int(std::floor(max_w_f));
    int max_h = int(std::floor(max_h_f));
    if (max_h % 2 == 1) max_h -= 1;
    const double beta = std::min(double(max_w * p * r) / width, double(max_h * p * r) / height);
    best_w = int(std::floor(width * beta / p)) * p;
    best_h = int(std::floor(height * beta / p)) * p;
  }
  const Grid g = grid_tokens(best_h, best_w, p, r);
  return {g.n_llm_h, g.n_llm_w, best_h, best_w, g.num_tokens};
}

// Python round(): half to even.
int py_round(double x) {
  const double f = std::floor(x), d = x - f;
  if (d > 0.5) return int(f) + 1;
  if (d < 0.5) return int(f);
  return int(f) % 2 == 0 ? int(f) : int(f) + 1;
}

}  // namespace

Patches preprocess(const RgbImage& img, const VisionConfig& cfg) {
  const int p = cfg.patch, r = cfg.downsample;
  if (img.w <= 0 || img.h <= 0) throw ImageError("image: empty");
  int width = img.w, height = img.h;
  if (cfg.max_wh_ratio > 0 && width > height * cfg.max_wh_ratio) width = height * cfg.max_wh_ratio;
  if (0 < double(width) * height && double(width) * height < cfg.min_pixels) {
    const double ratio = std::pow(double(cfg.min_pixels) / (double(width) * height), 0.5);
    width = int(width * ratio);
    height = int(height * ratio);
  }
  int best_w = int(std::ceil(double(width) / p)) * p, best_h = int(std::ceil(double(height) / p)) * p;
  // safe_resize: shrink the target until the block fits the token budget (minus 3 for alignment).
  const int max_n = cfg.max_tokens - 3;
  Grid g = grid_tokens(best_h, best_w, p, r);
  int n_llm_h = g.n_llm_h, n_llm_w = g.n_llm_w, num = g.num_tokens, budget = max_n;
  while (num > max_n) {
    const Fit f = solve_resize_ratio(height, width, p, r, budget);
    n_llm_h = f.n_llm_h;
    n_llm_w = f.n_llm_w;
    best_h = f.best_h;
    best_w = f.best_w;
    num = f.num_tokens;
    budget -= 1;
  }
  Patches out;
  out.n_vit_h = best_h / p;
  out.n_vit_w = best_w / p;
  out.n_llm_h = n_llm_h;
  out.n_llm_w = n_llm_w;
  RgbImage canvas;
  if (cfg.max_wh_ratio > 0 && img.w >= cfg.max_wh_ratio * img.h) {
    canvas = resize_bicubic(img, best_w, best_h);
  } else {
    // ImageOps.pad: contain() (aspect-preserving resize), centred on grey 127.
    int cw = best_w, chh = best_h;
    const double im_ratio = double(img.w) / img.h, dest_ratio = double(best_w) / best_h;
    if (im_ratio != dest_ratio) {
      if (im_ratio > dest_ratio) {
        const int nh = py_round(double(img.h) / img.w * best_w);
        if (nh != best_h) chh = nh;
      } else {
        const int nw = py_round(double(img.w) / img.h * best_h);
        if (nw != best_w) cw = nw;
      }
    }
    const RgbImage resized = resize_bicubic(img, cw, chh);
    if (cw == best_w && chh == best_h) {
      canvas = resized;
    } else {
      canvas = RgbImage{best_w, best_h, std::vector<uint8_t>(size_t(best_w) * best_h * 3, 127)};
      int ox = 0, oy = 0;
      if (cw != best_w) ox = py_round((best_w - cw) * 0.5);
      else oy = py_round((best_h - chh) * 0.5);
      for (int y = 0; y < chh; ++y) {
        const int ty = y + oy;
        if (ty < 0 || ty >= best_h) continue;
        for (int x = 0; x < cw; ++x) {
          const int tx = x + ox;
          if (tx < 0 || tx >= best_w) continue;
          std::memcpy(&canvas.px[(size_t(ty) * best_w + size_t(tx)) * 3], &resized.px[(size_t(y) * cw + size_t(x)) * 3], 3);
        }
      }
    }
  }
  // x / 255 in fp32, (x - 0.5) / 0.5, to bf16 (nearest even); patch (vh, vw) holds [c][py][px].
  uint16_t lut[256];
  for (int v = 0; v < 256; ++v) {
    const float x = (float(v) / 255.0f - 0.5f) / 0.5f;
    uint32_t u;
    std::memcpy(&u, &x, 4);
    lut[v] = uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
  out.data.resize(size_t(out.n_vit_h) * out.n_vit_w * 3 * p * p);
  size_t o = 0;
  for (int vh = 0; vh < out.n_vit_h; ++vh)
    for (int vw = 0; vw < out.n_vit_w; ++vw)
      for (int c = 0; c < 3; ++c)
        for (int py = 0; py < p; ++py)
          for (int px = 0; px < p; ++px)
            out.data[o++] = lut[canvas.px[(size_t(vh * p + py) * best_w + size_t(vw * p + px)) * 3 + size_t(c)]];
  return out;
}

ImageBlock build_image_block(int n_llm_h, int n_llm_w, int start_pos) {
  const int compress_pad = 3 - start_pos % 4;
  const int pad_h = n_llm_h % 2, rows = n_llm_h + pad_h, row_len = n_llm_w + 1;
  const int pad_last = rows / 2 * row_len % 2 * 2;
  // Grid types row-major ([kImage] * w + [kImageNewLine]) * h, then a padding row when h is odd.
  std::vector<int32_t> grid;
  for (int y = 0; y < n_llm_h; ++y) {
    for (int x = 0; x < n_llm_w; ++x) grid.push_back(kImage);
    grid.push_back(kImageNewLine);
  }
  for (int i = 0; i < row_len * pad_h; ++i) grid.push_back(kImagePad);
  // N-layout: rows in pairs, column by column (row 2i, row 2i + 1) -> 2 x 2 squares per 4 tokens.
  ImageBlock b;
  for (int i = 0; i < compress_pad; ++i) b.types.push_back(kImagePad);
  b.types.push_back(kImageStart);
  for (int pair = 0; pair < rows / 2; ++pair)
    for (int x = 0; x < row_len; ++x)
      for (int dy = 0; dy < 2; ++dy) {
        const int y = pair * 2 + dy, idx = y * row_len + x;
        b.types.push_back(grid[size_t(idx)]);
        if (y < n_llm_h && x < n_llm_w) b.perm.push_back(y * n_llm_w + x);
      }
  for (int i = 0; i < pad_last; ++i) b.types.push_back(kImagePad);
  b.types.push_back(kImageEnd);
  return b;
}

std::vector<uint8_t> image_bytes_from_url(const std::string& url) {
  if (url.rfind("data:", 0) != 0)
    throw ImageError("only data: URLs are supported for images (base64); fetch remote images client-side");
  const size_t comma = url.find(',');
  if (comma == std::string::npos || url.substr(0, comma).find(";base64") == std::string::npos)
    throw ImageError("unsupported data URL encoding (base64 is)");
  static const auto table = [] {
    std::array<int8_t, 256> t{};
    t.fill(-1);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) t[uint8_t(a[i])] = int8_t(i);
    t[uint8_t('-')] = 62;  // url-safe variants
    t[uint8_t('_')] = 63;
    return t;
  }();
  std::vector<uint8_t> out;
  out.reserve((url.size() - comma) * 3 / 4);
  uint32_t acc = 0;
  int bits = 0;
  for (size_t i = comma + 1; i < url.size(); ++i) {
    const unsigned char ch = static_cast<unsigned char>(url[i]);
    if (ch == '=') break;
    if (ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t') continue;
    const int v = table[ch];
    if (v < 0) throw ImageError("invalid base64 in data URL");
    acc = (acc << 6) | uint32_t(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(uint8_t((acc >> bits) & 0xff));
    }
  }
  return out;
}

}  // namespace unholy::vision

namespace unholy::vision {

std::vector<ImageInput> expand_images(std::vector<int32_t>& ids, int32_t placeholder, int32_t vocab,
                                      const std::vector<std::string>& urls, const VisionConfig& cfg) {
  const size_t n = size_t(std::count(ids.begin(), ids.end(), placeholder));
  if (n != urls.size())
    throw ImageError("found " + std::to_string(n) + " image tokens but got " + std::to_string(urls.size()) + " images");
  std::vector<ImageInput> out;
  if (n == 0) return out;
  std::vector<int32_t> expanded;
  expanded.reserve(ids.size() + n * size_t(cfg.max_tokens));
  size_t next = 0;
  for (const int32_t t : ids) {
    if (t != placeholder) {
      expanded.push_back(t);
      continue;
    }
    const std::vector<uint8_t> bytes = image_bytes_from_url(urls[next++]);
    ImageInput in;
    uint64_t h = 1469598103934665603ull;
    for (const uint8_t b : bytes) h = (h ^ b) * 1099511628211ull;
    in.hash = h;
    in.patches = preprocess(decode_image(bytes), cfg);
    in.start = int(expanded.size());
    in.block = build_image_block(in.patches.n_llm_h, in.patches.n_llm_w, in.start);
    for (const int32_t type : in.block.types) expanded.push_back(vocab + type);
    out.push_back(std::move(in));
  }
  ids = std::move(expanded);
  return out;
}

}  // namespace unholy::vision

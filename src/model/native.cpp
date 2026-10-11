#include "model/native.hpp"

#include <stdexcept>

#include "core/half.hpp"

namespace unholy::model {

void native_dense_cpu(const Checkpoint& ck, const std::string& prefix, float* w, int* out_features,
                      int* in_features) {
  const TensorInfo& wt = ck.at(prefix + ".weight");
  const TensorInfo& st = ck.at(prefix + ".scale");
  if (wt.shape.size() != 2 || st.shape.size() != 2) throw std::runtime_error(prefix + ": not 2-D");
  const auto sc = ck.as<uint8_t>(st);
  if (wt.dtype == DType::I8 && st.dtype == DType::F8_E8M0) {
    const int out = int(wt.shape[0]), in = int(wt.shape[1] * 2);
    if (st.shape[0] != out || st.shape[1] * 32 != in) throw std::runtime_error(prefix + ": MXFP4 scale shape");
    const auto q = ck.as<uint8_t>(wt);
    for (int o = 0; o < out; ++o)
      for (int i = 0; i < in; ++i) {
        const uint8_t byte = q[size_t(o) * (in / 2) + i / 2];
        const float v = e2m1_to_float((i & 1) ? (byte >> 4) : (byte & 0xf));
        w[size_t(o) * in + i] = v * e8m0_to_float(sc[size_t(o) * (in / 32) + i / 32]);
      }
    *out_features = out;
    *in_features = in;
    return;
  }
  if (wt.dtype == DType::F8_E4M3 && st.dtype == DType::F8_E8M0) {
    const int out = int(wt.shape[0]), in = int(wt.shape[1]);
    const int sb = int(st.shape[1]);
    if (st.shape[0] != (out + 127) / 128 || sb != (in + 127) / 128)
      throw std::runtime_error(prefix + ": fp8 scale shape");
    const auto q = ck.as<uint8_t>(wt);
    for (int o = 0; o < out; ++o)
      for (int i = 0; i < in; ++i)
        w[size_t(o) * in + i] = e4m3_to_float(q[size_t(o) * in + i]) * e8m0_to_float(sc[size_t(o / 128) * sb + i / 128]);
    *out_features = out;
    *in_features = in;
    return;
  }
  throw std::runtime_error(prefix + ": unsupported native format " + dtype_name(wt.dtype) + "/" +
                           dtype_name(st.dtype));
}

}  // namespace unholy::model

// Native DeepSeek-V4 checkpoint formats (the `deepseek-ai` safetensors release): CPU reference
// decoders the GPU kernels are tested against.
//
//   routed experts  experts.E.w{1,2,3}.weight  I8 [out, in/2]  MXFP4: two e2m1 per byte, low nibble
//                   first; .scale F8_E8M0 [out, in/32] (one power-of-two scale per 32 inputs)
//   dense fp8       *.weight F8_E4M3 [out, in]; .scale F8_E8M0 [out/128, in/128] (128x128 blocks)
#pragma once

#include <string>

#include "model/safetensors.hpp"

namespace unholy::model {

// Dense weight [out][in] in float from an MXFP4 expert or an fp8 block-scaled linear at `prefix`
// (".weight" / ".scale" appended). Throws on any other format.
void native_dense_cpu(const Checkpoint& ck, const std::string& prefix, float* w, int* out_features,
                      int* in_features);

}  // namespace unholy::model

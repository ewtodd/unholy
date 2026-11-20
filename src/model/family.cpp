#include "model/family.hpp"

#include <stdexcept>
#include <string>

namespace unholy::model {

Family detect_family(const gufo::json::Value& config) {
  const std::string type = config.member_str("model_type");
  if (type == "deepseek_v4") return Family::Dsv4;
  if (type == "glm5_next") return Family::Glm5Next;
  if (type == "hy_v4_flash") return Family::Hy4Flash;
  throw std::runtime_error("unsupported checkpoint: model_type '" + type + "' (supported: deepseek_v4, glm5_next, hy_v4_flash)");
}

const char* family_name(Family f) {
  switch (f) {
    case Family::Dsv4: return "DeepSeek-V4";
    case Family::Glm5Next: return "GLM-5.3-Flash";
    case Family::Hy4Flash: return "Hy4 flash";
  }
  return "?";
}

}  // namespace unholy::model

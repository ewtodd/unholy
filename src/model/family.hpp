// Which model a checkpoint holds, from its config.json, so the serving binary can pick the family's
// config, planner, step model and chat format.
#pragma once
#include "gufo/json.hpp"

namespace unholy::model {

enum class Family {
  Dsv4,      // DeepSeek-V4-Flash (and -Vision-Exp): model_type deepseek_v4
  Glm5Next,  // GLM-5.3-Flash: model_type glm5_next
};

// Throws for a model_type no family covers.
Family detect_family(const gufo::json::Value& config);
const char* family_name(Family f);

}  // namespace unholy::model

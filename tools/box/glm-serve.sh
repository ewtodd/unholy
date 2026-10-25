#!/usr/bin/env bash
# GLM-5.3-Flash (build B) served with the GLM service's configuration: tools/box/model-serve.sh glm.
#   tools/box/glm-serve.sh LABEL CMD...
exec "$(dirname "$0")/model-serve.sh" glm "$@"

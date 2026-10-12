cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1201 gfx1151)
endif()

project(unholy LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# No C++20 modules; without this CMake 4 wants clang-scan-deps for every try_compile.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(hip REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
# cpp-httplib, header-only, without TLS (its CMake config would pull in OpenSSL).
pkg_check_modules(HTTPLIB REQUIRED IMPORTED_TARGET httplib)

add_compile_options(-Wall -Wextra -Wno-unused-parameter)
# therock-sdk-nix's clang wrapper appends its link flags to compile-only invocations too.
add_compile_options(-Wno-unused-command-line-argument)

# Host-only pieces (no device code): sysfs topology, checkpoints, CPU references.
add_library(unholy_host STATIC
  src/hw/pci.cpp
  src/model/safetensors.cpp
  src/model/exl3.cpp
  src/model/native.cpp
)
target_include_directories(unholy_host PUBLIC src third_party)

add_executable(unholy-ckpt tools/ckpt/main.cpp)
target_link_libraries(unholy-ckpt PRIVATE unholy_host)

# unholy-probe: topology, verified transports, signalling latency.
add_executable(unholy-probe tools/probe/probe.hip)
set_source_files_properties(tools/probe/probe.hip PROPERTIES LANGUAGE HIP)
target_link_libraries(unholy-probe PRIVATE unholy_host hip::host)

# Serving text layer: JSON, tokenizer (PCRE2 pre-tokenizer), DSV4 chat format. No device code.
add_library(unholy_text STATIC
  src/text/json.cpp
  src/text/tokenizer.cpp
  src/text/chat.cpp
)
target_include_directories(unholy_text PUBLIC src)
target_link_libraries(unholy_text PUBLIC PkgConfig::PCRE2)

add_executable(unholy-text-test tools/text_test/main.cpp)
target_link_libraries(unholy-text-test PRIVATE unholy_text)

# Scheduler / sampler: host-only, tested against a fake model.
add_library(unholy_sched STATIC
  src/engine/engine.cpp
  src/engine/sampler.cpp
)
target_include_directories(unholy_sched PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(unholy_sched PUBLIC Threads::Threads)

add_executable(unholy-sched-test tools/sched_test/main.cpp)
target_link_libraries(unholy-sched-test PRIVATE unholy_sched)

# Reference kernels + the DSV4 reference model path.
add_library(unholy_ref STATIC
  src/kernels/ref_gemm.hip
  src/kernels/ref_elem.hip
  src/kernels/ref_attn.hip
  src/kernels/ref_moe.hip
  src/model/dsv4.hip
  src/model/forward_ref.hip
  src/model/loader.hip
  src/kernels/ref_exl3.hip
  src/kernels/exl3_moe.hip
  src/kernels/gemm_wmma.hip
  src/kernels/attn_wmma.hip
  src/kernels/hc_fast.hip
  src/kernels/moe_dispatch.hip
  src/kernels/gemv.hip
  src/kernels/indexer.hip
  src/kernels/mx_moe.hip
  src/kernels/dspark.hip
)
target_include_directories(unholy_ref PUBLIC src third_party)
target_link_libraries(unholy_ref PUBLIC unholy_host hip::host)

add_executable(unholy-layer-test tools/layer_test/main.hip)
target_link_libraries(unholy-layer-test PRIVATE unholy_ref)

install(TARGETS unholy-probe unholy-ckpt unholy-layer-test RUNTIME DESTINATION bin)

add_executable(unholy-ref-run tools/ref_run/main.hip)
target_link_libraries(unholy-ref-run PRIVATE unholy_ref)
install(TARGETS unholy-ref-run RUNTIME DESTINATION bin)

add_executable(unholy-batch-test tools/batch_test/main.hip)
target_link_libraries(unholy-batch-test PRIVATE unholy_ref)

# Engine: placement, resident multi-device model.
add_library(unholy_engine STATIC
  src/engine/plan.cpp
  src/engine/resident.hip
)
set_source_files_properties(src/engine/plan.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(unholy_engine PUBLIC src third_party)
target_link_libraries(unholy_engine PUBLIC unholy_ref)

add_executable(unholy-resident-test tools/resident_test/main.hip)
target_link_libraries(unholy-resident-test PRIVATE unholy_engine)

# The server: OpenAI-compatible HTTP frontend + the `unholy` binary.
add_library(unholy_server STATIC src/server/openai.cpp)
target_include_directories(unholy_server PUBLIC src)
target_link_libraries(unholy_server PUBLIC unholy_text unholy_sched PkgConfig::HTTPLIB)

add_executable(unholy tools/unholy/main.cpp)
set_source_files_properties(tools/unholy/main.cpp PROPERTIES LANGUAGE HIP)
target_compile_definitions(unholy PRIVATE UNHOLY_DATA_DIR="${CMAKE_INSTALL_PREFIX}/share/unholy")
target_link_libraries(unholy PRIVATE unholy_server unholy_engine)
install(TARGETS unholy unholy-text-test unholy-sched-test RUNTIME DESTINATION bin)
install(FILES data/expert_profile_dsv4engine_code32k.txt data/expert_profile_unholy_mixed.txt
              data/expert_profile_unholy_code32k.txt DESTINATION share/unholy)

add_executable(unholy-bench tools/bench/main.hip)
target_link_libraries(unholy-bench PRIVATE unholy_engine)

add_executable(unholy-indexer-test tools/kernel_test/indexer.hip)
target_link_libraries(unholy-indexer-test PRIVATE unholy_ref)

add_executable(unholy-exl3-gemm-test tools/kernel_test/exl3_gemm.hip)
target_link_libraries(unholy-exl3-gemm-test PRIVATE unholy_ref)

add_executable(unholy-curve tools/bench/curve.hip)
target_link_libraries(unholy-curve PRIVATE unholy_engine)

add_executable(unholy-dspark-test tools/kernel_test/dspark.hip)
target_link_libraries(unholy-dspark-test PRIVATE unholy_ref)
add_executable(unholy-mx-test tools/kernel_test/mx_moe.hip)
target_link_libraries(unholy-mx-test PRIVATE unholy_ref)

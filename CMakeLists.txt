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

install(TARGETS unholy-probe unholy-ckpt RUNTIME DESTINATION bin)

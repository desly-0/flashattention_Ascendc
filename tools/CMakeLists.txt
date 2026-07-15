# ==============================================================================
# FlashAttention V2 — Ascend C 算子独立开发构建
# CANN 8.5.0 / dav-2201 (Ascend 910B)
#
# 使用方法 (CPU 仿真模式，无需 NPU):
#   source ${ASCEND_HOME_PATH}/set_env.sh
#   cmake -S . -B build -DCMAKE_ASC_RUN_MODE=cpu
#   cmake --build build -j$(nproc)
#
# 或直接用: bash build.sh sim
# ==============================================================================

cmake_minimum_required(VERSION 3.16)

# ── 架构配置 ──
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING
    "NPU architecture: dav-2201 (Ascend910B)")

# ── 查找 CANN ASC 工具链 ──
find_package(ASC REQUIRED)

project(flash_attention_v2 LANGUAGES ASC CXX)

# ── 核函数编译 ──
add_library(flash_attention_v2_kernel OBJECT ../src/flash_attention.asc)
set_target_properties(flash_attention_v2_kernel PROPERTIES LINKER_LANGUAGE ASC)
target_compile_options(flash_attention_v2_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_include_directories(flash_attention_v2_kernel PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

message(STATUS "")
message(STATUS "FlashAttention V2 核函数开发构建")
message(STATUS "  architecture    : ${CMAKE_ASC_ARCHITECTURES}")
message(STATUS "  run mode        : ${CMAKE_ASC_RUN_MODE}")
message(STATUS "  ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")
message(STATUS "  构建目标: flash_attention_v2_kernel")
message(STATUS "")

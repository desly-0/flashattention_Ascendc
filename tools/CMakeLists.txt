# ==============================================================================
# FlashAttention Ascend C 算子 — 独立开发构建
#
# 专注于核函数开发 (Ascend C .asc → .o)
# Host 端算子注册在部署阶段由 CANN 框架完成
#
# 使用方法 (CPU 仿真模式，无需 NPU):
#   source /usr/local/Ascend/cann-9.0.0/set_env.sh
#   cmake -S . -B build -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201
#   cmake --build build -j$(nproc)
#
# 或直接用: bash build.sh sim
# ==============================================================================

cmake_minimum_required(VERSION 3.16)

# 默认使用 dav-2201 (Ascend910B1/Ascend910B2/B3) 架构
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING
    "NPU architecture: dav-2201 (Ascend910B), dav-3510 (Ascend 950)")

# 查找 CANN ASC 工具链 — 提供 ASC 语言支持
find_package(ASC REQUIRED)

project(flash_attention_kernel LANGUAGES ASC CXX)

# ============================================================================
# FlashAttention 核函数
#   .asc → .o  (bisheng 编译器)
#   CPU 模式: --run-mode=cpu  → x86_64 原生 ELF，可 CPU 仿真
#   NPU 模式: 默认 → dav-2201 目标二进制
# ============================================================================
add_library(flash_attention_kernel OBJECT op_kernel/flash_attention.asc)
set_target_properties(flash_attention_kernel PROPERTIES LINKER_LANGUAGE ASC)
target_compile_options(flash_attention_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_include_directories(flash_attention_kernel PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)

# ============================================================================
# 在 NPU 上部署时，需额外编译 host 端算子注册
# 操作步骤:
#   1. cd op_host
#   2. 将 flash_attention.cpp 放入完整算子项目的 op_host/ 目录
#   3. 参考原项目 op_host/CMakeLists.txt 与 npu_op_code_gen 配合编译
# ============================================================================
message(STATUS "")
message(STATUS "FlashAttention 核函数开发构建")
message(STATUS "  architecture    : ${CMAKE_ASC_ARCHITECTURES}")
message(STATUS "  run mode        : ${CMAKE_ASC_RUN_MODE}")
message(STATUS "  compiler        : ${CMAKE_ASC_COMPILER}")
message(STATUS "  构建目标: flash_attention_kernel")
message(STATUS "")

#!/bin/bash
# ==============================================================================
# FlashAttention V2 — Ascend C 算子构建脚本
# CANN 8.5.0 / dav-2201 (Ascend 910B)
#
# 用法:
#   bash build.sh            # CPU 仿真模式构建（默认，无需 NPU）
#   bash build.sh npu        # NPU 设备模式构建（需要 NPU）
#   bash build.sh clean      # 清理构建目录
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
MODE="${1:-sim}"

# 1. 加载 CANN 环境
if [ -z "$ASCEND_HOME_PATH" ]; then
    # 尝试多个可能的 CANN 安装路径
    for CANDIDATE in /home/hello/Ascend/cann-8.5.0/set_env.sh \
                     /usr/local/Ascend/cann-8.5.0/set_env.sh \
                     /usr/local/Ascend/cann-9.0.0/set_env.sh; do
        if [ -f "$CANDIDATE" ]; then
            source "$CANDIDATE"
            echo "[INFO] 已加载: $CANDIDATE"
            break
        fi
    done
fi

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "[ERROR] ASCEND_HOME_PATH 未设置，且找不到 set_env.sh"
    echo "请先 source 您的 CANN 环境变量"
    exit 1
fi
echo "[INFO] ASCEND_HOME_PATH = $ASCEND_HOME_PATH"

# 2. 处理构建模式
case "$MODE" in
    clean)
        echo "[INFO] 清理构建目录..."
        rm -rf "$BUILD_DIR"
        echo "[INFO] 已清理"
        exit 0
        ;;
    sim|cpu)
        echo "[INFO] CPU 仿真模式 (无需 NPU)"
        echo "[INFO] 编译为 x86_64 原生 .o，可 CPU 仿真执行"
        EXTRA_FLAGS="-DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_BUILD_TYPE=Debug"
        ;;
    npu|device)
        echo "[INFO] NPU 设备模式"
        echo "[INFO] 编译为 dav-2201 (Ascend910B) 目标"
        EXTRA_FLAGS="-DCMAKE_BUILD_TYPE=Release"
        ;;
    *)
        echo "用法: bash build.sh [sim|npu|clean]"
        echo "  sim   — CPU 仿真 (默认，无需 NPU)"
        echo "  npu   — NPU 设备"
        echo "  clean — 清理"
        exit 1
        ;;
esac

# 3. CMake 配置
echo ""
echo "[INFO] 配置 CMake ..."
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_ASC_ARCHITECTURES="dav-2201" \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    $EXTRA_FLAGS \
    2>&1

# 4. 编译
echo ""
echo "[INFO] 编译中 ..."
cmake --build "$BUILD_DIR" -j$(nproc) 2>&1

# 5. 结果展示
KERNEL_O="$BUILD_DIR/CMakeFiles/flash_attention_v2_kernel.dir/__/src/flash_attention.asc.o"
if [ ! -f "$KERNEL_O" ]; then
    KERNEL_O="$BUILD_DIR/CMakeFiles/flash_attention_v2_kernel.dir/src/flash_attention.asc.o"
fi

echo ""
echo "==============================================="
echo " [SUCCESS] 构建完成！"
echo "==============================================="
echo ""
if [ -f "$KERNEL_O" ]; then
    echo "  编译产物: $(basename "$KERNEL_O")"
    ls -lh "$KERNEL_O"
    echo "  文件类型: $(file -b "$KERNEL_O")"
else
    echo "  ⚠ 未找到核函数 .o 文件"
    echo "  搜索路径:"
    find "$BUILD_DIR" -name "*.o" 2>/dev/null | head -5
fi
echo ""
echo "  下一步:"
echo "    python3 ../test/test_ref.py        运行参考实现测试"
echo "    bash run_test.sh                   运行仿真测试"
echo ""

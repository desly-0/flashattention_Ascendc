#!/bin/bash
# ==============================================================================
# FlashAttention Ascend C 算子 — 构建脚本
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
    if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
        source /usr/local/Ascend/cann-9.0.0/set_env.sh
    else
        echo "[ERROR] ASCEND_HOME_PATH 未设置，且找不到 set_env.sh"
        echo "请先 source 您的 CANN 环境变量"
        exit 1
    fi
fi
echo "[INFO] ASCEND_HOME_PATH = $ASCEND_HOME_PATH"

# 2. 处理模式
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

# 5. 结果
KERNEL_O="$BUILD_DIR/CMakeFiles/flash_attention_kernel.dir/op_kernel/flash_attention.asc.o"
echo ""
echo "==============================================="
echo " [SUCCESS] 构建完成！"
echo "==============================================="
echo ""
if [ -f "$KERNEL_O" ]; then
    echo "  编译产物: flash_attention.asc.o"
    ls -lh "$KERNEL_O"
    echo "  文件类型: $(file -b "$KERNEL_O")"
else
    echo "  ⚠ 未找到核函数 .o 文件，请检查构建日志"
fi
echo ""
echo "  下一步:"
echo "    python3 test_ref.py    运行参考实现测试"
echo "    bash run_test.sh       运行仿真测试"
echo ""

#!/bin/bash
# ==============================================================================
# FlashAttention V2 — 测试运行脚本
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo " FlashAttention V2 Test Suite"
echo "================================================"
echo ""

# 1. 检查 CANN 环境
if [ -z "$ASCEND_HOME_PATH" ]; then
    for CANDIDATE in /home/hello/Ascend/cann-8.5.0/set_env.sh \
                     /usr/local/Ascend/cann-8.5.0/set_env.sh; do
        if [ -f "$CANDIDATE" ]; then
            source "$CANDIDATE"
            echo "[INFO] 已加载: $CANDIDATE"
            break
        fi
    done
fi
echo "[INFO] ASCEND_HOME_PATH = $ASCEND_HOME_PATH"
echo ""

# 2. 运行参考测试
echo "--- 步骤 1: 参考实现验证 (Python) ---"
python3 "$PROJECT_DIR/test/test_ref.py" \
    --batch 1 --heads-q 4 --heads-kv 1 --seq-q 256 --seq-k 256 --dim 64 \
    --causal 2>&1
echo ""

# 3. 构建核函数 (CPU 仿真模式)
echo "--- 步骤 2: 构建核函数 (CPU 仿真) ---"
bash "$SCRIPT_DIR/build.sh" sim 2>&1
echo ""

# 4. 检查编译产物
KERNEL_O="$SCRIPT_DIR/build/CMakeFiles/flash_attention_v2_kernel.dir/__/src/flash_attention.asc.o"
if [ ! -f "$KERNEL_O" ]; then
    KERNEL_O="$SCRIPT_DIR/build/CMakeFiles/flash_attention_v2_kernel.dir/src/flash_attention.asc.o"
fi

if [ -f "$KERNEL_O" ]; then
    echo "--- 步骤 3: 核函数编译成功 ---"
    ls -lh "$KERNEL_O"
    file "$KERNEL_O"
else
    echo "--- 步骤 3: ⚠ 核函数 .o 未找到 ---"
    echo "  可能的原因: 编译模式不是 CPU 仿真，或编译错误"
fi

echo ""
echo "================================================"
echo " 测试完成"
echo "================================================"
echo ""
echo "NPU 测试 (需要 NPU 硬件):"
echo "  python3 $PROJECT_DIR/test/test_npu.py"
echo ""
echo "端到端测试:"
echo "  python3 $PROJECT_DIR/test/test_e2e.py"
echo ""

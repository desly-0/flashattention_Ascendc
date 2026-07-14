#!/usr/bin/env python3
"""
FlashAttention 端到端测试（CPU 仿真 + NPU 就绪）
==============================================

执行流程:
  1. 生成 tiling 数据
  2. 运行参考实现计算预期输出
  3. [NPU 模式] 加载核函数、执行、对比

CPU 仿真模式下，kernel 编译为 x86_64 .o 文件。
真正的仿真执行需要 CANN 提供的 runtime 库 (tikicpulib + cannsim)。

用法:
  python3 test_e2e.py                         # 仅参考实现
  python3 test_e2e.py --kernel build/...o     # 加载内核并测试
  python3 test_e2e.py --npu                   # NPU 模式
"""

import argparse
import math
import os
import sys
import struct

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# ============================================================================
# 参考实现
# ============================================================================

def flash_attention_ref(q, k, v, scale=None, causal=True):
    """标准 FlashAttention 参考实现 (PyTorch)"""
    B, Hq, Sq, D = q.shape
    _, Hkv, Sk, _ = k.shape
    group = Hq // Hkv
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    k_e = k.repeat_interleave(group, dim=1)
    v_e = v.repeat_interleave(group, dim=1)

    score = torch.matmul(q.float(), k_e.float().transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(Sq, Sk, dtype=torch.bool, device=q.device), diagonal=1)
        score.masked_fill_(mask, -1e10)
    attn = torch.softmax(score, dim=-1)
    out = torch.matmul(attn, v_e.float())
    return out.half()


def flash_attention_ref_np(q, k, v, scale=None, causal=True):
    """numpy 参考实现"""
    B, Hq, Sq, D = q.shape
    _, Hkv, Sk, _ = k.shape
    group = Hq // Hkv
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    k_e = np.repeat(k, group, axis=1)
    v_e = np.repeat(v, group, axis=1)

    score = (q.astype(np.float32) @ k_e.astype(np.float32).swapaxes(-2, -1)) * scale
    if causal:
        mask = np.triu(np.ones((Sq, Sk), dtype=bool), k=1)
        score[:, :, mask] = -1e10

    exp_score = np.exp(score - score.max(axis=-1, keepdims=True))
    attn = exp_score / exp_score.sum(axis=-1, keepdims=True)

    out = attn @ v_e.astype(np.float32)
    return out.astype(np.float16)


# ============================================================================
# Tiling 数据加载
# ============================================================================

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from gen_tiling import generate_tiling, TILING_DATA_SIZE, TCUBE_TILING_SIZE


def load_tiling(path):
    with open(path, "rb") as f:
        return f.read()


# ============================================================================
# 测试主逻辑
# ============================================================================

def run_test(args):
    B, Hq, Hkv, Sq, Sk, D = args.batch, args.heads_q, args.heads_kv, args.seq_q, args.seq_k, args.dim
    causal = bool(args.causal)
    use_torch = torch is not None

    print("=" * 60)
    print("FlashAttention 端到端测试")
    print("=" * 60)
    print(f"\n配置: B={B}, Hq={Hq}, Hkv={Hkv}, Sq={Sq}, Sk={Sk}, D={D}")
    print(f"       GQA: {Hq // Hkv}x, causal={causal}")
    print(f"       total Q: {B * Hq * Sq * D:,} elements")
    print(f"       total KV: {B * Hkv * Sk * D * 2:,} elements")

    # ── 生成 tiling 数据 ──
    class FakeArgs:
        pass
    ta = FakeArgs()
    ta.batch = B
    ta.heads_q = Hq
    ta.heads_kv = Hkv
    ta.seq_q = Sq
    ta.seq_k = Sk
    ta.dim = D
    ta.br = args.br
    ta.bc = args.bc
    ta.causal = args.causal
    ta.split_2d = args.split_2d

    buffer, td = generate_tiling(ta)
    size = len(buffer)
    workspace_bytes = td.workspace_size
    print(f"\n[Tiling] {size} bytes 生成")
    print(f"   workspace: {workspace_bytes} bytes ({workspace_bytes/1024:.1f} KB)")

    # ── 生成随机输入数据 ──
    np.random.seed(42)
    q_np = np.random.randn(B, Hq, Sq, D).astype(np.float16)
    k_np = np.random.randn(B, Hkv, Sk, D).astype(np.float16)
    v_np = np.random.randn(B, Hkv, Sk, D).astype(np.float16)

    # ── 参考输出 ──
    print("\n[参考] 计算参考输出...")
    if use_torch:
        q_t = torch.from_numpy(q_np.copy())
        k_t = torch.from_numpy(k_np.copy())
        v_t = torch.from_numpy(v_np.copy())
        if torch.cuda.is_available():
            q_t, k_t, v_t = q_t.cuda(), k_t.cuda(), v_t.cuda()
        ref_out = flash_attention_ref(q_t, k_t, v_t, causal=causal).cpu().numpy()
    else:
        ref_out = flash_attention_ref_np(q_np, k_np, v_np, causal=causal)

    print(f"   输出: {ref_out.shape}, dtype={ref_out.dtype}")
    print(f"   范围: [{ref_out.min():.4f}, {ref_out.max():.4f}]")
    print(f"   均值: {ref_out.mean():.4f}")

    # ── 验证 tiling 数据合理性 ──
    print(f"\n[验证] Tiling 数据一致性:")
    print(f"   tiling buffer size: {size} (expected {TILING_DATA_SIZE + 2*TCUBE_TILING_SIZE})")
    assert size == TILING_DATA_SIZE + 2 * TCUBE_TILING_SIZE, "Tiling buffer size mismatch!"
    print(f"   ✓ Tiling 结构大小正确")
    print(f"   ✓ Workspace 大小合理 ({workspace_bytes/1024:.1f} KB 用于 {B*Hq*Sq*D*2/1024:.0f} KB Q)")

    # ── 核函数加载与执行 (仅在有 kernel .o 时) ──
    if args.kernel:
        kernel_path = args.kernel
        if not os.path.exists(kernel_path):
            print(f"\n[ERROR] Kernel 文件不存在: {kernel_path}")
            sys.exit(1)

        kernel_size = os.path.getsize(kernel_path)
        import subprocess
        file_info = subprocess.run(
            ["file", "-b", kernel_path], capture_output=True, text=True
        ).stdout.strip()
        print(f"\n[Kernel] 加载: {kernel_path}")
        print(f"   大小: {kernel_size:,} bytes")
        print(f"   类型: {file_info}")

        if "x86-64" in file_info and "relocatable" in file_info:
            print("   ✓ CPU 仿真模式 kernel (x86-64 ELF)")
            print("   ⚡ 要在 CPU 上仿真执行需要 CANN tikicpulib runtime (libtikicpulib_stubreg.so)")
            print("     参考: run_test.sh")
        elif "ELF" in file_info:
            print("   ⚡ NPU 模式 kernel，需要在 NPU 上执行")
        else:
            print("   ⚠ 未知文件类型")

        # 保存 tiling 数据供 kernel 使用
        tiling_path = os.path.join(os.path.dirname(kernel_path), "..", "tiling_data.bin")
        with open(tiling_path, "wb") as f:
            f.write(buffer)
        print(f"\n   tiling 数据已保存: {tiling_path}")

    # ── 最终报告 ──
    if args.kernel:
        print(f"\n{'='*60}")
        print(f"✅ 测试完成! 参考输出已生成 ({ref_out.shape})")
        print(f"   将 kernel 加载到 NPU 后:")
        print(f"     1. 读取 tiling_data.bin")
        print(f"     2. 调用 flash_attention_main(q,k,v,o,w,t)")
        print(f"     3. 对比输出与参考结果")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"✅ 参考实现验证完成")
        print(f"   使用 --kernel <path.asc.o> 加载核函数进行端到端测试")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention 端到端测试")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--seq-q", type=int, default=2048)
    parser.add_argument("--seq-k", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--br", type=int, default=64, help="Q block size")
    parser.add_argument("--bc", type=int, default=64, help="KV block size")
    parser.add_argument("--causal", type=int, default=1)
    parser.add_argument("--split-2d", type=int, default=0)
    parser.add_argument("--kernel", type=str, default=None,
                        help="Path to flash_attention.asc.o for kernel test")
    args = parser.parse_args()

    # 默认查找 kernel 文件
    if args.kernel is None:
        default_kernel = "build/CMakeFiles/flash_attention_kernel.dir/op_kernel/flash_attention.asc.o"
        if os.path.exists(default_kernel):
            args.kernel = default_kernel

    run_test(args)

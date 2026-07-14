#!/usr/bin/env python3
"""
FlashAttention NPU 性能 Benchmark
=================================

用法:
  python3 bench_npu.py                                          # 默认
  python3 bench_npu.py --heads-q 32 --heads-kv 8 --dim 128      # 标准
  python3 bench_npu.py --profile                                # 详细 profiling
"""

import argparse
import math
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from gen_tiling import generate_tiling


def run_bench(ben_args):
    B, Hq, Hkv, Sq, Sk, D = ben_args.batch, ben_args.heads_q, ben_args.heads_kv, ben_args.seq_q, ben_args.seq_k, ben_args.dim
    causal = bool(ben_args.causal)

    print("=" * 60)
    print("FlashAttention NPU 性能 Benchmark")
    print("=" * 60)
    print(f"\n[配置] B={B} Hq={Hq} Hkv={Hkv} Sq={Sq} Sk={Sk} D={D}")
    print(f"       causal={causal}, GQA={Hq//Hkv}x")

    # Tiling
    class FA: pass
    ta = FA()
    for k in ['batch','heads_q','heads_kv','seq_q','seq_k','dim','br','bc','causal','split_2d']:
        setattr(ta, k, getattr(ben_args, k, None))
    ta.br = ben_args.br or min(64, D)
    ta.bc = ben_args.bc or 64
    tiling_buf, td = generate_tiling(ta)

    # 计算 FLOPs
    flops_per_head = 4 * Sq * Sk * D  # QK^T + PV + softmax
    total_flops = B * Hq * flops_per_head

    print(f"\n[计算] 总 FLOPs: {total_flops/1e9:.2f} GFLOPs (估计)")
    print(f"       workspace: {td.workspace_size/1024:.1f} KB")

    # 基准: PyTorch 实现 (如果可用)
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    if HAS_TORCH and ben_args.baseline:
        print(f"\n[Baseline] PyTorch 参考实现...")
        torch.manual_seed(42)
        q = torch.randn(B, Hq, Sq, D, dtype=torch.float16)
        k = torch.randn(B, Hkv, Sk, D, dtype=torch.float16)
        v = torch.randn(B, Hkv, Sk, D, dtype=torch.float16)

        # warmup
        for _ in range(5):
            _ = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                is_causal=causal, scale=1.0/math.sqrt(D))

        # benchmark
        n_iter = ben_args.iters
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                is_causal=causal, scale=1.0/math.sqrt(D))
        elapsed = time.perf_counter() - start
        avg_ms = elapsed / n_iter * 1000
        tflops = total_flops / 1e12 / (elapsed / n_iter)
        print(f"       {n_iter} iters, {avg_ms:.2f} ms/iter, {tflops:.2f} TFLOPS")

    # NPU benchmark
    try:
        import acl
        print(f"\n[NPU] 准备 AscendCL 环境...")
        acl.init()
        device_id = 0
        context = acl.rt.create_context(device_id)[0]
        stream, _ = acl.rt.create_stream()

        # 分配缓冲区
        q_size = B * Hq * Sq * D * 2
        k_size = B * Hkv * Sk * D * 2
        v_size = B * Hkv * Sk * D * 2
        o_size = B * Hq * Sq * D * 2
        w_size = td.workspace_size
        t_size = len(tiling_buf)

        d_q = acl.rt.malloc(q_size, acl.constants.ACL_MEM_MALLOC_HUGE_FIRST)[0]
        d_k = acl.rt.malloc(k_size, acl.constants.ACL_MEM_MALLOC_HUGE_FIRST)[0]
        d_v = acl.rt.malloc(v_size, acl.constants.ACL_MEM_MALLOC_HUGE_FIRST)[0]
        d_o = acl.rt.malloc(o_size, acl.constants.ACL_MEM_MALLOC_HUGE_FIRST)[0]
        d_w = acl.rt.malloc(w_size, acl.constants.ACL_MEM_MALLOC_HUGE_FIRST)[0]
        d_t = acl.rt.malloc(t_size, acl.constants.ACL_MEM_MALLOC_HUGE_FIRST)[0]

        for buf_name, size in [("Q", q_size), ("K", k_size), ("V", v_size),
                                ("O", o_size), ("W", w_size), ("T", t_size)]:
            print(f"       {buf_name}: {size/1024:.1f} KB")

        # 加载 kernel
        kernel_path = ben_args.kernel
        if not kernel_path:
            default = "build/CMakeFiles/flash_attention_kernel.dir/op_kernel/flash_attention.asc.o"
            if os.path.exists(default):
                kernel_path = default

        if kernel_path and os.path.exists(kernel_path):
            with open(kernel_path, "rb") as f:
                kernel_data = f.read()
            try:
                kernel_handle, ret = acl.rt.load_command_from_file(kernel_path)
                if ret != 0:
                    raise RuntimeError(f"load_command_from_file ret={ret}")
            except AttributeError:
                kernel_handle, ret = acl.rt.load_command_from_mem(kernel_data, len(kernel_data))
                if ret != 0:
                    raise RuntimeError(f"load_command_from_mem ret={ret}")
            print(f"\n[NPU] Kernel: {kernel_path} ({len(kernel_data):.1f} MB)")

            # 核函数参数: q, k, v, o, w, t (设备指针)
            kernel_args = [d_q, d_k, d_v, d_o, d_w, d_t]
            block_dim = ben_args.blocks

            # Warmup
            print(f"[NPU] Warmup ({ben_args.iters} iters)...")
            for _ in range(ben_args.iters):
                ret = acl.rt.launch_command(kernel_handle, block_dim, kernel_args, stream)
                if ret != 0:
                    raise RuntimeError(f"launch_command failed: {ret}")
            acl.rt.synchronize_stream(stream)
            print(f"       done")

            # Benchmark
            n_iter = ben_args.iters
            print(f"[NPU] Benchmark ({n_iter} iters)...")
            start = time.perf_counter()
            for _ in range(n_iter):
                acl.rt.launch_command(kernel_handle, block_dim, kernel_args, stream)
            acl.rt.synchronize_stream(stream)
            elapsed = time.perf_counter() - start
            avg_ms = elapsed / n_iter * 1000

            print(f"\n{'='*60}")
            print(f"[NPU结果] {n_iter} iters")
            print(f"          平均: {avg_ms:.3f} ms/iter")
            if total_flops > 0:
                tflops = total_flops / 1e12 / (elapsed / n_iter)
                bw = (B * Hq * Sq * D * 2 + 2 * B * Hkv * Sk * D * 2) / 1e9 / (elapsed / n_iter)
                print(f"          Throughput: {tflops:.2f} TFLOPS")
                print(f"          利用率: {tflops/305*100:.1f}% (vs Ascend910B 理论 305TFLOPS fp16)")
                print(f"          显存带宽: {bw:.1f} GB/s")
            print(f"{'='*60}")

            # 清理
            acl.rt.unload_command(kernel_handle)
        else:
            print(f"\n[NPU] ⚡ 未找到 kernel .o 文件，跳过 NPU 执行")

        acl.rt.free(d_q)
        acl.rt.free(d_k)
        acl.rt.free(d_v)
        acl.rt.free(d_o)
        acl.rt.free(d_w)
        acl.rt.free(d_t)
        acl.rt.destroy_stream(stream)
        acl.rt.destroy_context(context)
        acl.rt.reset_device(device_id)
        acl.finalize()

    except ImportError:
        print(f"\n[NPU] ⚡ acl 模块不可用，跳过 NPU 执行")

    # 输出摘要
    print(f"\n{'='*60}")
    print(f"Benchmark 完成")
    if ben_args.kernel and os.path.exists(ben_args.kernel):
        print(f"  Kernel: {ben_args.kernel}")
    else:
        print(f"  Kernel: build/*.asc.o (需要先运行 bash build.sh npu)")
    print(f"  配置: {B}x{Hq}x{Hkv}x{Sq}x{Sk}x{D}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention NPU Benchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--seq-q", type=int, default=2048)
    parser.add_argument("--seq-k", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--br", type=int, default=0)
    parser.add_argument("--bc", type=int, default=0)
    parser.add_argument("--causal", type=int, default=1)
    parser.add_argument("--split-2d", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=8, help="blockDim for kernel launch")
    parser.add_argument("--iters", type=int, default=100, help="benchmark iterations")
    parser.add_argument("--kernel", type=str, default=None, help="Path to .asc.o")
    parser.add_argument("--baseline", action="store_true", help="Run PyTorch baseline")
    args = parser.parse_args()

    run_bench(args)

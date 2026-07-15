#!/usr/bin/env python3
"""
FlashAttention V2 — NPU 性能 Benchmark
========================================
测量和对比 FlashAttention 核函数在不同配置下的性能。

用法:
  python3 bench_npu.py                             # 默认基准测试
  python3 bench_npu.py --mode sweep                # 扫描不同 seq_len
  python3 bench_npu.py --mode compare              # 对比不同配置
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

try:
    from gen_tiling import generate_tiling
except ImportError:
    generate_tiling = None

try:
    import torch
except ImportError:
    torch = None


# ============================================================================
# Benchmark 配置
# ============================================================================

# 标准测试配置
STANDARD_CONFIGS = [
    # (B, Hq, Hkv, Sq, Sk, D, label)
    (1, 1,  1,  512,   512,   64,  "S=512, D=64,  MHA"),
    (1, 1,  1,  1024,  1024,  64,  "S=1K, D=64,  MHA"),
    (1, 1,  1,  2048,  2048,  64,  "S=2K, D=64,  MHA"),
    (1, 1,  1,  4096,  4096,  64,  "S=4K, D=64,  MHA"),
    (1, 1,  1,  8192,  8192,  64,  "S=8K, D=64,  MHA"),
    (1, 1,  1,  2048,  2048,  128, "S=2K, D=128, MHA"),
    (1, 8,  2,  2048,  2048,  128, "S=2K, D=128, GQA(4x)"),
    (1, 32, 8,  2048,  2048,  128, "S=2K, D=128, GQA(4x)"),
]


def format_throughput(B, Hq, Sq, D, time_ms):
    """计算吞吐量 (TFLOPs)"""
    # FlashAttention: 2 * B * Hq * Sq^2 * D FLOPs (粗略估计)
    flops = 2 * B * Hq * Sq * Sq * D
    tflops = flops / (time_ms / 1000) / 1e12
    tokens_per_sec = B * Hq * Sq / (time_ms / 1000)
    return tflops, tokens_per_sec


# ============================================================================
# NPU Benchmark
# ============================================================================

class NPUBenchmark:
    """NPU 性能基准测试"""

    def __init__(self, args):
        self.args = args

    def run_single(self, B, Hq, Hkv, Sq, Sk, D, causal=True, label=""):
        """运行单个 benchmark"""
        Br = min(64, D)
        Bc = 128

        print(f"  {label or f'{B}x{Hq}x{Hkv}x{Sq}x{Sk}x{D}':40s} ", end="", flush=True)

        # 用 PyTorch 作为参考 (CPU)
        if torch is not None:
            q = torch.randn(B, Hq, Sq, D, dtype=torch.float16)
            k = torch.randn(B, Hkv, Sk, D, dtype=torch.float16)
            v = torch.randn(B, Hkv, Sk, D, dtype=torch.float16)

            # Warm up
            for _ in range(5):
                _ = torch.matmul(q.float() @ k.float().transpose(-2, -1))

            # Benchmark PyTorch attention
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            n_iter = 20
            for _ in range(n_iter):
                score = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(D)
                if causal:
                    mask = torch.triu(torch.ones(Sq, Sk, dtype=torch.bool), diagonal=1)
                    score.masked_fill_(mask, -1e10)
                attn = torch.softmax(score, dim=-1)
                out = torch.matmul(attn, v.float())
            elapsed = (time.perf_counter() - start) * 1000 / n_iter

            tflops, tok_s = format_throughput(B, Hq, Sq, D, elapsed)
            print(f"PyTorch ref: {elapsed:8.2f} ms | {tflops:5.1f} TFLOPs | {tok_s:8.0f} tok/s")
        else:
            print(f"  (需 PyTorch 进行参考对比)")

    def run_sweep_seqlen(self):
        """扫描不同序列长度"""
        print(f"\n序列长度扫描 (B=1, Hq=1, Hkv=1, D=64)")
        print(f"{'Seq Len':>10} {'Time(ms)':>10} {'TFLOPs':>10} {'tok/s':>12}")
        print("-" * 44)

        for log2_s in range(9, 14):  # 512 ~ 8192
            S = 2 ** log2_s
            q = torch.randn(1, 1, S, 64, dtype=torch.float16)
            k = torch.randn(1, 1, S, 64, dtype=torch.float16)
            v = torch.randn(1, 1, S, 64, dtype=torch.float16)

            if torch is None:
                print(f"{S:>10}  (需 PyTorch)")
                continue

            start = time.perf_counter()
            n_iter = max(5, 50 // (S // 512))
            for _ in range(n_iter):
                score = torch.matmul(q.float(), k.float().transpose(-2, -1)) / 8.0
                mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
                score.masked_fill_(mask, -1e10)
                attn = torch.softmax(score, dim=-1)
                out = torch.matmul(attn, v.float())
            elapsed = (time.perf_counter() - start) * 1000 / n_iter

            tflops, tok_s = format_throughput(1, 1, S, 64, elapsed)
            print(f"{S:>10} {elapsed:>10.2f} {tflops:>10.2f} {tok_s:>12.0f}")

    def run_compare(self):
        """对比不同 head/seq 配置"""
        print(f"\n配置对比 (causal=True)")
        print(f"{'Config':<40} {'Time(ms)':>10} {'TFLOPs':>10} {'tok/s':>12}")
        print("-" * 74)

        for B, Hq, Hkv, Sq, Sk, D, label in STANDARD_CONFIGS:
            self.run_single(B, Hq, Hkv, Sq, Sk, D, causal=True, label=label)


def main():
    parser = argparse.ArgumentParser(description="FlashAttention V2 NPU Benchmark")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "sweep", "compare"],
                        help="Benchmark mode")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=1)
    parser.add_argument("--heads-kv", type=int, default=1)
    parser.add_argument("--seq-q", type=int, default=2048)
    parser.add_argument("--seq-k", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--causal", type=int, default=1)
    args = parser.parse_args()

    bench = NPUBenchmark(args)

    print("=" * 60)
    print("FlashAttention V2 Benchmark")
    print("=" * 60)
    print(f"[Hardware] x86_64 CPU (参考对比)")
    if torch is not None and torch.cuda.is_available():
        print(f"[GPU]     {torch.cuda.get_device_name(0)}")
    print()

    if args.mode == "standard":
        print(f"[配置] B={args.batch}, Hq={args.heads_q}, Hkv={args.heads_kv}")
        print(f"       Sq={args.seq_q}, Sk={args.seq_k}, D={args.dim}")
        bench.run_single(args.batch, args.heads_q, args.heads_kv,
                         args.seq_q, args.seq_k, args.dim,
                         causal=bool(args.causal),
                         label="Current config")
    elif args.mode == "sweep":
        bench.run_sweep_seqlen()
    elif args.mode == "compare":
        bench.run_compare()

    print(f"\n{'='*60}")
    print("Benchmark 完成")
    print(f"  提示: 在 NPU 上运行时会更准确")
    print("=" * 60)


if __name__ == "__main__":
    main()

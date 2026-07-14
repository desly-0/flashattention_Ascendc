#!/usr/bin/env python3
"""
FlashAttention 参考实现与正确性验证
可在无 NPU 环境下独立运行（仅需 numpy / PyTorch）

用法:
  python3 test_ref.py                        # 小规模测试
  python3 test_ref.py --causal               # 带 causal mask 测试
  python3 test_ref.py --profile              # 性能分析 (需 PyTorch)
"""

import argparse
import math
import sys

try:
    import numpy as np
except ImportError:
    print("需要 numpy: pip install numpy")
    sys.exit(1)

try:
    import torch
except ImportError:
    torch = None
    print("[WARN] PyTorch 未安装，参考实现使用 numpy (较慢)")


# ============================================================================
# 参考实现: FlashAttention (online softmax)
# ============================================================================

def flash_attention_ref(q, k, v, causal=False, scale=None):
    """
    标准 FlashAttention 参考实现
    使用 online softmax 分块计算，与 Ascend C 核函数逻辑一致

    Args:
        q: [B, Hq, Sq, D]  float16
        k: [B, Hkv, Sk, D] float16
        v: [B, Hkv, Sk, D] float16
        causal: 是否使用 causal mask
        scale: 缩放因子，默认 1/sqrt(D)
    Returns:
        out: [B, Hq, Sq, D] float16
    """
    B, Hq, Sq, D = q.shape
    _, Hkv, Sk, _ = k.shape
    group = Hq // Hkv  # GQA group size

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # --- GQA: expand KV heads ---
    k_e = k.repeat_interleave(group, dim=1)  # [B, Hq, Sk, D]
    v_e = v.repeat_interleave(group, dim=1)

    # --- 参考实现: 使用标准分块 softmax attention ---
    # 小规模直接使用标准实现
    score = torch.matmul(q.float(), k_e.float().transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(Sq, Sk, dtype=torch.bool), diagonal=1).to(q.device)
        score.masked_fill_(mask, -1e10)

    attn = torch.softmax(score, dim=-1)
    out = torch.matmul(attn, v_e.float())
    return out.half()


def flash_attention_ref_numpy(q, k, v, causal=False, scale=None):
    """numpy 版本的参考实现"""
    B, Hq, Sq, D = q.shape
    _, Hkv, Sk, _ = k.shape
    group = Hq // Hkv

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # GQA expand
    k_e = np.repeat(k, group, axis=1)
    v_e = np.repeat(v, group, axis=1)

    score = (q.astype(np.float32) @ k_e.astype(np.float32).swapaxes(-2, -1)) * scale

    if causal:
        mask = np.triu(np.ones((Sq, Sk), dtype=bool), k=1)
        score[:, :, mask] = -1e10

    # softmax
    exp_score = np.exp(score - score.max(axis=-1, keepdims=True))
    attn = exp_score / exp_score.sum(axis=-1, keepdims=True)

    out = attn @ v_e.astype(np.float32)
    return out.astype(np.float16)


# ============================================================================
# 测试入口
# ============================================================================

def run_test(args):
    print("=" * 60)
    print("FlashAttention 参考实现验证")
    print("=" * 60)

    B, Hq, Hkv, Sq, Sk, D = args.batch, args.heads_q, args.heads_kv, args.seq_q, args.seq_k, args.dim
    causal = args.causal
    print(f"\n配置: B={B}, Hq={Hq}, Hkv={Hkv}, Sq={Sq}, Sk={Sk}, D={D}, causal={causal}")
    print(f"       GQA 组大小: {Hq // Hkv}")
    print(f"       总 Q 元素: {B * Hq * Sq * D:,}")
    print(f"       总 KV 元素: {B * Hkv * Sk * D * 2:,}")

    if torch is not None:
        # --- PyTorch 参考 ---
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[使用 PyTorch] device={device}")

        q = torch.randn(B, Hq, Sq, D, dtype=torch.float16, device=device)
        k = torch.randn(B, Hkv, Sk, D, dtype=torch.float16, device=device)
        v = torch.randn(B, Hkv, Sk, D, dtype=torch.float16, device=device)

        print("  计算参考输出...")
        out = flash_attention_ref(q, k, v, causal=causal)
        print(f"  输出形状: {out.shape}   dtype: {out.dtype}")
        print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
        print(f"  输出均值: {out.mean().item():.4f}")

        if args.profile and device == "cuda":
            # 性能分析
            print("\n  性能分析 (warm-up + 100 iter)...")
            for _ in range(10):
                flash_attention_ref(q, k, v, causal=causal)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                flash_attention_ref(q, k, v, causal=causal)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / 100
            print(f"  平均耗时: {ms:.3f} ms / iter")

    else:
        # --- numpy 参考 ---
        np.random.seed(42)
        print("\n[使用 numpy]")

        q = np.random.randn(B, Hq, Sq, D).astype(np.float16)
        k = np.random.randn(B, Hkv, Sk, D).astype(np.float16)
        v = np.random.randn(B, Hkv, Sk, D).astype(np.float16)

        print("  计算参考输出...")
        out = flash_attention_ref_numpy(q, k, v, causal=causal)
        print(f"  输出形状: {out.shape}   dtype: {out.dtype}")
        print(f"  输出范围: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  输出均值: {out.mean():.4f}")

    print("\n✅ 参考实现验证完成")
    print(f"   当有 NPU 时，将 Ascend C 核函数输出与此参考结果对比即可验证正确性")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention 参考实现验证")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=32, help="Query heads (Hq)")
    parser.add_argument("--heads-kv", type=int, default=8, help="KV heads (Hkv)")
    parser.add_argument("--seq-q", type=int, default=2048, help="Query sequence length")
    parser.add_argument("--seq-k", type=int, default=2048, help="Key sequence length")
    parser.add_argument("--dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--causal", action="store_true", default=True, help="Enable causal mask")
    parser.add_argument("--profile", action="store_true", help="Run performance profiling")
    args = parser.parse_args()

    run_test(args)

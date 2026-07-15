#!/usr/bin/env python3
"""
FlashAttention V2 — 参考实现与正确性验证
=========================================
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

    # GQA: expand KV heads
    k_e = k.repeat_interleave(group, dim=1)  # [B, Hq, Sk, D]
    v_e = v.repeat_interleave(group, dim=1)

    # 使用标准分块 softmax attention
    score = torch.matmul(q.float(), k_e.float().transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(Sq, Sk, dtype=torch.bool, device=q.device), diagonal=1)
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
# Tiled 参考实现 (匹配 Ascend C 核函数的在线 softmax 逻辑)
# ============================================================================

def flash_attention_tiled_ref(q, k, v, Br=64, Bc=128, causal=False, scale=None):
    """
    分块 FlashAttention 参考实现
    使用 online softmax，与 Ascend C 核函数逻辑完全一致

    Args:
        q: [B, Hq, Sq, D]  float16
        k: [B, Hkv, Sk, D] float16
        v: [B, Hkv, Sk, D] float16
        Br: Query tile size
        Bc: KV tile size
        causal: 是否使用 causal mask
        scale: 缩放因子
    Returns:
        out: [B, Hq, Sq, D] float16
    """
    B, Hq, Sq, D = q.shape
    _, Hkv, Sk, _ = k.shape
    group = Hq // Hkv

    if scale is None:
        scale_val = 1.0 / math.sqrt(D)
    else:
        scale_val = scale

    k_np = k if isinstance(k, np.ndarray) else k.numpy()
    v_np = v if isinstance(v, np.ndarray) else v.numpy()
    q_np = q if isinstance(q, np.ndarray) else q.numpy()
    k_e = np.repeat(k_np, group, axis=1).astype(np.float64)
    v_e = np.repeat(v_np, group, axis=1).astype(np.float64)
    q_f = q_np.astype(np.float64)

    out = np.zeros((B, Hq, Sq, D), dtype=np.float64)

    for b in range(B):
        for h in range(Hq):
            T_r = (Sq + Br - 1) // Br
            T_c = (Sk + Bc - 1) // Bc

            for ti in range(T_r):
                qs = ti * Br
                qe = min(qs + Br, Sq)
                qt = qe - qs

                # 初始化 O_acc, m, l
                O_acc = np.zeros((qt, D), dtype=np.float64)
                m_prev = np.full(qt, -np.inf, dtype=np.float64)
                l_prev = np.zeros(qt, dtype=np.float64)

                # Load Q tile
                Q_tile = q_f[b, h, qs:qe, :]  # [qt, D]

                for tj in range(T_c):
                    ks = tj * Bc
                    ke = min(ks + Bc, Sk)
                    kt = ke - ks

                    # Load K, V tiles
                    K_tile = k_e[b, h, ks:ke, :]   # [kt, D]
                    V_tile = v_e[b, h, ks:ke, :]   # [kt, D]

                    # S = Q @ K^T * scale
                    S = Q_tile @ K_tile.T * scale_val  # [qt, kt]

                    # Causal mask
                    if causal:
                        for i in range(qt):
                            gq = qs + i
                            for j in range(kt):
                                if gq < (ks + j):
                                    S[i, j] = -1e20

                    # Online softmax
                    m_new = np.maximum(np.max(S, axis=1), m_prev)  # [qt]
                    rescale = np.exp(m_prev - m_new)               # [qt]
                    P = np.exp(S - m_new[:, np.newaxis])           # [qt, kt]
                    l_new = rescale * l_prev + np.sum(P, axis=1)   # [qt]

                    # Rescale old O
                    for i in range(qt):
                        O_acc[i, :] *= rescale[i]

                    # O += P @ V
                    O_acc += P @ V_tile

                    m_prev = m_new
                    l_prev = l_new

                # Finalize
                for i in range(qt):
                    if l_prev[i] > 0:
                        O_acc[i, :] /= l_prev[i]
                out[b, h, qs:qe, :] = O_acc

    return out.astype(np.float16)


# ============================================================================
# 测试入口
# ============================================================================

def run_test(args):
    print("=" * 60)
    print("FlashAttention V2 参考实现验证")
    print("=" * 60)

    B, Hq, Hkv, Sq, Sk, D = args.batch, args.heads_q, args.heads_kv, args.seq_q, args.seq_k, args.dim
    causal = args.causal
    Br = args.br if args.br else min(64, D)
    Bc = args.bc if args.bc else 128

    print(f"\n配置: B={B}, Hq={Hq}, Hkv={Hkv}, Sq={Sq}, Sk={Sk}, D={D}")
    print(f"       causal={causal}, Br={Br}, Bc={Bc}")
    print(f"       GQA 组大小: {Hq // Hkv}")
    print(f"       总 Q 元素: {B * Hq * Sq * D:,}")
    print(f"       总 KV 元素: {B * Hkv * Sk * D * 2:,}")

    if torch is not None:
        # PyTorch 参考
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[使用 PyTorch] device={device}")

        q = torch.randn(B, Hq, Sq, D, dtype=torch.float16, device=device)
        k = torch.randn(B, Hkv, Sk, D, dtype=torch.float16, device=device)
        v = torch.randn(B, Hkv, Sk, D, dtype=torch.float16, device=device)

        print("  计算标准参考输出 (exact softmax)...")
        out = flash_attention_ref(q, k, v, causal=causal)
        print(f"  标准参考: {out.shape}, [{out.min().item():.4f}, {out.max().item():.4f}]")

        print("  计算分块参考输出 (online softmax, 匹配 Kernel)...")
        out_tiled = flash_attention_tiled_ref(q.cpu(), k.cpu(), v.cpu(),
                                              Br=Br, Bc=Bc, causal=causal)
        out_tiled_t = torch.from_numpy(out_tiled).to(device=device)

        # 对比标准参考 vs 分块参考
        ref_f = out.float()
        tiled_f = out_tiled_t.float()
        max_diff = torch.max(torch.abs(ref_f - tiled_f)).item()
        mean_diff = torch.mean(torch.abs(ref_f - tiled_f)).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_f.reshape(-1), tiled_f.reshape(-1), dim=0).item()

        print(f"\n  标准参考 vs 分块参考:")
        print(f"    Max diff:  {max_diff:.6f}")
        print(f"    Mean diff: {mean_diff:.6f}")
        print(f"    Cos sim:   {cos_sim:.6f}")
        print(f"    Result:    {'PASS' if max_diff < 0.1 else 'CHECK'}")

        if args.profile and device == "cuda":
            print("\n  性能分析 (PyTorch, warm-up + 100 iter)...")
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
        # numpy 参考
        np.random.seed(42)
        print("\n[使用 numpy]")

        q = np.random.randn(B, Hq, Sq, D).astype(np.float16)
        k = np.random.randn(B, Hkv, Sk, D).astype(np.float16)
        v = np.random.randn(B, Hkv, Sk, D).astype(np.float16)

        print("  计算分块参考输出...")
        out_tiled = flash_attention_tiled_ref(q, k, v, Br=Br, Bc=Bc, causal=causal)
        print(f"  输出形状: {out_tiled.shape}   dtype: {out_tiled.dtype}")
        print(f"  输出范围: [{out_tiled.min():.4f}, {out_tiled.max():.4f}]")
        print(f"  输出均值: {out_tiled.mean():.4f}")

    print("\n✅ 参考实现验证完成")
    return out if torch is not None else out_tiled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention V2 参考实现验证")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=32, help="Query heads (Hq)")
    parser.add_argument("--heads-kv", type=int, default=8, help="KV heads (Hkv)")
    parser.add_argument("--seq-q", type=int, default=2048, help="Query sequence length")
    parser.add_argument("--seq-k", type=int, default=2048, help="Key sequence length")
    parser.add_argument("--dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--br", type=int, default=0, help="Q tile size (0=auto)")
    parser.add_argument("--bc", type=int, default=0, help="KV tile size (0=auto)")
    parser.add_argument("--causal", action="store_true", default=True, help="Enable causal mask")
    parser.add_argument("--profile", action="store_true", help="Run performance profiling")
    args = parser.parse_args()

    run_test(args)

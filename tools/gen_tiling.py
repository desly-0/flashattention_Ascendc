#!/usr/bin/env python3
"""
FlashAttention V2 — Tiling 数据生成器
======================================
为 CANN 8.5.0 / dav-2201 (Ascend 910B) 生成核函数所需的
FlashAttentionTilingData + 2×TCubeTiling 二进制数据。

用法:
  from gen_tiling import generate_tiling
  tiling_buf, td = generate_tiling(args)
  # tiling_buf: bytes that can be uploaded to device memory
"""

import argparse
import math
import struct
import sys

# ============================================================================
# 常量定义
# ============================================================================

# TCubeTiling: 50 个 int32_t 字段 (200 bytes)
# 参考: CANN 8.5 matmul_intf.h 中的 TCubeTiling 定义
TCUBE_TILING_SIZE = 50  # int32_t units
TCUBE_TILING_BYTES = TCUBE_TILING_SIZE * 4

# FlashAttentionTilingData: 15 个字段 (8-byte aligned, packed)
# batch, numHeads, numKvHeads, seqLenQ, seqLenK, headDim = 6 uint32
# scale = 1 float
# qBlockSize, kvBlockSize, headsPerBlock, isCausal, blockSplitMode,
#   workspaceSize, tilingQKOffset, tilingPVOffset = 9 uint32
# Total: (6 + 1 + 9) = 16 fields (1 float + 15 uint32) = 64 bytes
TILING_DATA_SIZE = 16  # fields (1 float + 15 uint32)
TILING_DATA_BYTES = TILING_DATA_SIZE * 4  # 64 bytes

# 总 tiling 数据大小
TILING_TOTAL_BYTES = TILING_DATA_BYTES + 2 * TCUBE_TILING_BYTES


# ============================================================================
# Tiling 计算
# ============================================================================

def compute_tiling(args):
    """
    根据输入参数计算最优 tiling 参数。

    Args:
        args: 包含如下属性的对象
            - batch, heads_q, heads_kv, seq_q, seq_k, dim
            - br (optional, default=min(64, dim)), bc (optional, default=64)
            - causal (0/1), split_2d (0/1)
    Returns:
        dict: tiling 参数
    """
    B = args.batch
    Hq = args.heads_q
    Hkv = args.heads_kv
    Sq = args.seq_q
    Sk = args.seq_k
    D = args.dim
    causal = getattr(args, 'causal', 1)
    split_2d = getattr(args, 'split_2d', 0)
    Br = getattr(args, 'br', 0) or min(64, D)
    Bc = getattr(args, 'bc', 0) or 64

    # 确保 tile size 合理
    if Br == 0 or Br > 64:
        Br = min(64, D)
    if Bc == 0 or Bc > 128:
        Bc = 64


    Br = min(Br, Sq)
    Bc = min(Bc, Sk)

    # Compute heads per core for UB estimation
    group_size = Hq // Hkv if Hkv > 0 else 1
    bs = min(group_size, Hq)

    # UB capacity check (Ascend 910B: ~256KB)
    # Major consumers: bS + bP = bs*Br*Bc*6, bO+bV = Br*D*8, bMask = Bc*D*4
    ub_est = (bs * Br * Bc * 6 + Br * D * 16 + Bc * D * 4 +
              32 * bs * Br + 5120)
    while ub_est > 250 * 1024 and (Br > 16 or Bc > 16):
        if Bc > Br and Bc > 32:
            Bc //= 2
        elif Br > 16:
            Br //= 2
        else:
            Bc //= 2
        ub_est = (bs * Br * Bc * 6 + Br * D * 16 + Bc * D * 4 +
                  32 * bs * Br + 5120)
    if ub_est > 250 * 1024:
        print(f"[WARN] UB estimate {ub_est/1024:.0f}KB may exceed limit (Br={Br}, Bc={Bc})")


    # 计算 head count per block (1D split)
    nQb = (Sq + Br - 1) // Br
    total_work = Hq * nQb

    # Workspace size (half units → bytes: *2)
    # P: Br*Bc halves
    # Pbatch: bs*Br*Bc halves
    # Qblock: bs*Br*D halves
    # S[0], S[1]: 2*bs*Br*Bc floats each (float=2 halves)
    # PV: bs*Br*D floats
    # O_acc: bs*Br*D floats
    # R0, R1: 2*bs*Br floats
    w_halfs = (Br * Bc) + (bs * Br * Bc) + (bs * Br * D) \
        + 2 * (2 * bs * Br * Bc) \
        + (bs * Br * D) * 2 \
        + (bs * Br * D) * 2 \
        + (bs * Br) * 2 \
        + (bs * Br) * 2
    # Add overhead safety margin
    w_halfs = int(w_halfs * 1.2 + 4096)
    workspace_size = w_halfs * 2  # bytes

    # TCubeTiling offsets (after FlashAttentionTilingData)
    tiling_qk_offset = TILING_DATA_BYTES
    tiling_pv_offset = TILING_DATA_BYTES + TCUBE_TILING_BYTES

    return {
        'batch': B,
        'numHeads': Hq,
        'numKvHeads': Hkv,
        'seqLenQ': Sq,
        'seqLenK': Sk,
        'headDim': D,
        'scale': 1.0 / math.sqrt(float(D)),
        'qBlockSize': Br,
        'kvBlockSize': Bc,
        'headsPerBlock': bs,
        'isCausal': causal,
        'blockSplitMode': split_2d,
        'workspaceSize': workspace_size,
        'tilingQKOffset': tiling_qk_offset,
        'tilingPVOffset': tiling_pv_offset,
        # Extended params (not in struct, but used by generation)
        'groupSize': group_size,
        'nQb': nQb,
    }


def generate_tcube_tiling_qk(td):
    """
    生成 bmm1 (Q×K^T) 的 TCubeTiling。

    Q: [bs*Br, D] half
    K: [1*Bc, D] half → K^T: [D, Bc]
    S = Q @ K^T: [bs*Br, Bc] float

    关键参数:
    - M = bs*Br, N = Bc, K = D
    - baseM = 32|16 (ref: tuning.md §2.2, cube fractal base)
    - baseN = 16 (cube base, always)
    - baseK = 16 (cube base, always)
    """
    Br = td['qBlockSize']
    Bc = td['kvBlockSize']
    D = td['headDim']
    bs = td['headsPerBlock']
    M = bs * Br
    N = Bc
    K = D

    tiling = [0] * TCUBE_TILING_SIZE

    # ── 基础形状 ──
    tiling[0] = M      # M
    tiling[1] = N      # N
    tiling[2] = K      # K

    # ── 分块参数 (ref: ascend-c-dev-8.5/ref/tuning.md §2.2) ──
    # CANN 8.5 tuning: baseM=32 (D>=128) or 16 (D<128); baseN=baseK=16
    tiling[3] = 32 if D >= 128 else 16   # baseM
    tiling[4] = 16                        # baseN (cube fractal)
    tiling[5] = 16                        # baseK (cube fractal)

    # ── 步长 ──
    tiling[6] = M      # stepM (full batch)
    tiling[7] = N      # stepN (full tile)
    tiling[8] = K      # stepK (full dim)

    # ── A 矩阵参数 (Q: [bs*Br, D]) ──
    tiling[9]  = 0     # aOrgRow (row offset)
    tiling[10] = 0     # aOrgCol (col offset)
    tiling[11] = K     # aStride (leading dim = D)

    # ── B 矩阵参数 (K: [Bc, D], transposed: K^T [D, Bc]) ──
    tiling[12] = 0     # bOrgRow
    tiling[13] = 0     # bOrgCol
    tiling[14] = K     # bStride (leading dim = D, since K is NT stored)

    # ── C 矩阵参数 (S: [bs*Br, Bc]) ──
    tiling[15] = 0     # cOrgRow
    tiling[16] = 0     # cOrgCol
    tiling[17] = N     # cStride (leading dim = Bc)

    # ── 格式参数 ──
    tiling[18] = 0     # aFormat (ND=0)
    tiling[19] = 0     # bFormat (ND=0)
    tiling[20] = 0     # cFormat (ND=0)

    # ── 数据类型 ──
    tiling[21] = 0     # aType (half=0 in CANN enum)
    tiling[22] = 0     # bType (half=0)
    tiling[23] = 1     # cType (float=1)

    # ── Cube 模式 ──
    tiling[24] = 0     # cubeMode
    tiling[25] = 1     # isBTranspose (K is transposed) ← key for Q@K^T
    tiling[26] = 0     # isATranspose

    # ── Batch ──
    tiling[27] = 0     # batch (no batching in cube tiling)

    # ── Depth (L1 分块) ──
    # CANN 8.5 optimization: bmm1 (Q×K^T) prefetches Q via L1 (depthA1=1)
    # Reference: ascend-c-dev-8.5/ref/compute_optimization.md §2.4
    tiling[30] = 1     # depthA1: L1 prefetch A matrix (Q)
    tiling[31] = 0     # depthB1
    tiling[32] = 0     # depthA2
    tiling[33] = 0     # depthB2

    # ── L1/L0 分块 (0 = auto by CANN runtime) ──
    tiling[34] = 0     # l0aSize
    tiling[35] = 0     # l0bSize
    tiling[36] = 0     # l1Size

    return tiling


def generate_tcube_tiling_pv(td):
    """
    生成 bmm2 (P×V) 的 TCubeTiling。

    P: [bs*Br, Bc] half
    V: [Bc, D] half
    PV = P @ V: [bs*Br, D] float

    关键参数:
    - M = bs*Br, N = D, K = Bc
    - baseM = 32|16 (ref: tuning.md §2.2, cube fractal base)
    - baseN = 16 (cube base, always)
    - baseK = 16 (cube base, always)
    """
    Br = td['qBlockSize']
    Bc = td['kvBlockSize']
    D = td['headDim']
    bs = td['headsPerBlock']
    M = bs * Br
    N = D
    K = Bc

    tiling = [0] * TCUBE_TILING_SIZE

    # ── 基础形状 ──
    tiling[0] = M      # M (= bs*Br)
    tiling[1] = N      # N (= D)
    tiling[2] = K      # K (= Bc)

    # ── 分块参数 (ref: ascend-c-dev-8.5/ref/tuning.md §2.2) ──
    # CANN 8.5 tuning: baseM=32 (D>=128) or 16 (D<128); baseN=baseK=16
    tiling[3] = 32 if D >= 128 else 16   # baseM
    tiling[4] = 16                        # baseN (cube fractal)
    tiling[5] = 16                        # baseK (cube fractal)

    # ── 步长 ──
    tiling[6] = M      # stepM
    tiling[7] = N      # stepN
    tiling[8] = K      # stepK

    # ── A 矩阵参数 (P: [bs*Br, Bc]) ──
    tiling[9]  = 0     # aOrgRow
    tiling[10] = 0     # aOrgCol
    tiling[11] = K     # aStride (= Bc)

    # ── B 矩阵参数 (V: [Bc, D], NT) ──
    tiling[12] = 0     # bOrgRow
    tiling[13] = 0     # bOrgCol
    tiling[14] = N     # bStride (= D)

    # ── C 矩阵参数 (PV: [bs*Br, D]) ──
    tiling[15] = 0     # cOrgRow
    tiling[16] = 0     # cOrgCol
    tiling[17] = N     # cStride (= D)

    # ── 格式参数 ──
    tiling[18] = 0     # aFormat (ND)
    tiling[19] = 0     # bFormat (ND)
    tiling[20] = 0     # cFormat (ND)

    # ── 数据类型 ──
    tiling[21] = 0     # aType (half)
    tiling[22] = 0     # bType (half)
    tiling[23] = 1     # cType (float)

    # ── Cube 模式 ──
    tiling[24] = 0     # cubeMode
    tiling[25] = 0     # isBTranspose (V is NOT transposed for P@V)
    tiling[26] = 0     # isATranspose

    # ── Batch ──
    tiling[27] = 0     # batch

    # ── Depth (L1 分块) ──
    # CANN 8.5 optimization: bmm2 (P×V) prefetches V via L1 (depthB1=1)
    # V is reused across Q blocks — L1 prefetch reduces L0B wait
    # Reference: ascend-c-dev-8.5/ref/compute_optimization.md §2.4
    tiling[30] = 0     # depthA1 (P is regenerated each block, no prefetch)
    tiling[31] = 1     # depthB1: L1 prefetch B matrix (V)
    tiling[32] = 0     # depthA2
    tiling[33] = 0     # depthB2

    return tiling


def pack_tiling_data(td, tcube_qk, tcube_pv):
    """
    将 tiling 数据结构打包为二进制 bytes。

    布局:
    [0..63]     FlashAttentionTilingData  (16 × uint32/float)
    [64..263]   TCubeTiling for bmm1       (50 × int32)
    [264..463]  TCubeTiling for bmm2       (50 × int32)
    总计: 464 bytes
    """
    buf = bytearray(TILING_TOTAL_BYTES)
    offset = 0

    # ── FlashAttentionTilingData ──
    fields = [
        ('I', td['batch']),
        ('I', td['numHeads']),
        ('I', td['numKvHeads']),
        ('I', td['seqLenQ']),
        ('I', td['seqLenK']),
        ('I', td['headDim']),
        ('f', td['scale']),
        ('I', td['qBlockSize']),
        ('I', td['kvBlockSize']),
        ('I', td['headsPerBlock']),
        ('I', td['isCausal']),
        ('I', td['blockSplitMode']),
        ('I', td['workspaceSize']),
        ('I', td['tilingQKOffset']),
        ('I', td['tilingPVOffset']),
    ]
    # 第16个字段（索引15）= 0 填充（对齐到8字节边界）
    # 实际上 FlashAttentionTilingData 有 16 个 uint32（含 scale 占一个位置）
    # 但 scale 是 float，所以总数还是 16*4 = 64 bytes
    # The struct has 15 fields + 1 padding = 16 uint32 equivalents
    for fmt, val in fields:
        struct.pack_into(fmt, buf, offset, val)
        offset += 4

    # Pad to 64 bytes (should already be there)
    while offset < TILING_DATA_BYTES:
        struct.pack_into('I', buf, offset, 0)
        offset += 4

    # ── TCubeTiling for bmm1 ──
    for v in tcube_qk:
        struct.pack_into('i', buf, offset, v)
        offset += 4

    # ── TCubeTiling for bmm2 ──
    for v in tcube_pv:
        struct.pack_into('i', buf, offset, v)
        offset += 4

    assert offset == TILING_TOTAL_BYTES, \
        f"Tiling data size mismatch: {offset} != {TILING_TOTAL_BYTES}"
    return bytes(buf)


def generate_tiling(args):
    """
    生成完整的 tiling 数据。

    Args:
        args: 包含 batch/heads_q/heads_kv/seq_q/seq_k/dim/br/bc/causal/split_2d 的对象
    Returns:
        (bytes, dict): tiling 二进制数据和 tiling 参数字典
    """
    td = compute_tiling(args)
    tcube_qk = generate_tcube_tiling_qk(td)
    tcube_pv = generate_tcube_tiling_pv(td)
    buf = pack_tiling_data(td, tcube_qk, tcube_pv)
    return buf, td


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FlashAttention V2 Tiling 生成器")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=4)
    parser.add_argument("--heads-kv", type=int, default=1)
    parser.add_argument("--seq-q", type=int, default=256)
    parser.add_argument("--seq-k", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--br", type=int, default=0, help="Q tile size (0=auto)")
    parser.add_argument("--bc", type=int, default=0, help="KV tile size (0=auto)")
    parser.add_argument("--causal", type=int, default=1)
    parser.add_argument("--split-2d", type=int, default=0)
    parser.add_argument("--output", type=str, default="", help="输出文件路径")
    args = parser.parse_args()

    tiling_buf, td = generate_tiling(args)

    print("=" * 60)
    print("FlashAttention V2 Tiling 参数")
    print("=" * 60)
    print(f"\n[输入]")
    print(f"  B={args.batch}, Hq={args.heads_q}, Hkv={args.heads_kv}")
    print(f"  Sq={args.seq_q}, Sk={args.seq_k}, D={args.dim}")
    print(f"  causal={args.causal}, split_2d={args.split_2d}")

    print(f"\n[Tiling 参数]")
    for k, v in td.items():
        print(f"  {k:20s} = {v}")

    print(f"\n[二进制数据]")
    print(f"  总大小: {len(tiling_buf)} bytes")
    print(f"  结构体: {TILING_DATA_BYTES} bytes")
    print(f"  TCubeTiling: {TCUBE_TILING_BYTES} bytes × 2")
    print(f"  Hex 前 64 bytes:")
    for i in range(0, min(64, len(tiling_buf)), 16):
        hex_str = ' '.join(f'{b:02x}' for b in tiling_buf[i:i+16])
        print(f"    [{i:04d}] {hex_str}")

    if args.output:
        with open(args.output, 'wb') as f:
            f.write(tiling_buf)
        print(f"\n  已保存: {args.output} ({len(tiling_buf)} bytes)")

    print(f"\n  Workspace 需求: {td['workspaceSize'] / 1024:.1f} KB")


if __name__ == "__main__":
    main()

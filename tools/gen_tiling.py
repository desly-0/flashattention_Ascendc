#!/usr/bin/env python3
"""
FlashAttention Tiling 数据生成器

生成 FlashAttentionTilingData + 2×TCubeTiling 的完整 tiling buffer。
kernel 通过 REGIST_MATMUL_OBJ 加载 tiling 数据来配置 Matmul 流水线。

用法:
  python3 gen_tiling.py                          # 默认参数，输出到 stdout
  python3 gen_tiling.py --out tiling.bin          # 写入文件
  python3 gen_tiling.py --print                   # 打印十六进制
  python3 gen_tiling.py --verify                  # 用 numpy 验证 workspace 合理性
"""
import argparse
import struct
import math
import sys
import os

# ============================================================================
# FlashAttentionTilingData (与 C 结构体严格对齐)
# pragma pack(push, 8)
# ============================================================================
TILING_STRUCT_FMT = "=IIIIIIfIIIIIIII"  # 14×uint32 + 1×float = 60 bytes

TILING_DATA_SIZE = struct.calcsize(TILING_STRUCT_FMT)


class TilingData:
    def __init__(self, batch=1, num_heads=32, num_kv_heads=8,
                 seq_q=2048, seq_k=2048, head_dim=128,
                 q_block_size=64, kv_block_size=64,
                 heads_per_block=1, is_causal=1, block_split_mode=0):
        self.batch = batch
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.seq_q = seq_q
        self.seq_k = seq_k
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.q_block_size = q_block_size
        self.kv_block_size = kv_block_size
        self.heads_per_block = heads_per_block
        self.is_causal = is_causal
        self.block_split_mode = block_split_mode
        self.workspace_size = self._calc_workspace_size()
        self.tiling_qk_offset = TILING_DATA_SIZE
        self.tiling_pv_offset = TILING_DATA_SIZE + TCUBE_TILING_SIZE

    def _calc_workspace_size(self):
        """计算 workspace 大小 (bytes)，与 kernel Init 中的布局一致"""
        Br = self.q_block_size
        Bc = self.kv_block_size
        D = self.head_dim
        bsz = self.num_heads // self.num_kv_heads
        if bsz < 1:
            bsz = 1

        # 所有偏移以 half (uint16) 为单位
        # V1 layout: R0(rescale) + R1(inv_l) after O — no overlap with O
        Poff = 0
        Pboff = Poff + Br * Bc
        Qboff = Pboff + bsz * Br * Bc
        Soff = Qboff + bsz * Br * D
        SV = Soff + 2 * bsz * Br * Bc * 2
        PVoff = SV + bsz * Br * D * 2
        Ooff = PVoff + bsz * Br * D * 2
        R0off = Ooff + bsz * Br * D * 2         # R0: bsz*Br floats after O
        R1off = R0off + bsz * Br * 2            # R0 = bsz*Br floats = bsz*Br*2 halfs
        wEnd = R1off + bsz * Br * 2             # R1 = bsz*Br floats = bsz*Br*2 halfs

        # wEnd 是 half 数量，转为 bytes
        return wEnd * 2  # half = 2 bytes

    def pack(self):
        """打包为 bytes"""
        return struct.pack(TILING_STRUCT_FMT,
            self.batch, self.num_heads, self.num_kv_heads,
            self.seq_q, self.seq_k,
            self.head_dim, self.scale,
            self.q_block_size, self.kv_block_size,
            self.heads_per_block, self.is_causal,
            self.block_split_mode, self.workspace_size,
            self.tiling_qk_offset, self.tiling_pv_offset)

    def __repr__(self):
        return (
            f"FlashAttentionTilingData(\n"
            f"  B={self.batch}, Hq={self.num_heads}, Hkv={self.num_kv_heads},\n"
            f"  Sq={self.seq_q}, Sk={self.seq_k}, D={self.head_dim},\n"
            f"  scale={self.scale:.6f},\n"
            f"  Br={self.q_block_size}, Bc={self.kv_block_size},\n"
            f"  headsPerBlock={self.heads_per_block},\n"
            f"  isCausal={self.is_causal}, splitMode={self.block_split_mode},\n"
            f"  workspaceSize={self.workspace_size} bytes,\n"
            f"  tilingQKOffset={self.tiling_qk_offset},\n"
            f"  tilingPVOffset={self.tiling_pv_offset},\n"
            f")")


# ============================================================================
# TCubeTiling (sizeof = 50× int32_t = 200 bytes, pack(8))
# 与 kernel_tiling.h 中的定义一致
# ============================================================================
TCUBE_TILING_SIZE = 200  # 50 int32_t fields

def make_tcube_tiling(m, n, k, single_m=None, single_n=None, single_k=None,
                      base_m=16, base_n=16, base_k=16,
                      used_cores=1, is_bmm1=False):
    """
    生成基本的 TCubeTiling 结构。

    对于 Ascend910B (dav-2201):
    - Cube base: 16×16×16
    - 推荐对齐到 16 的倍数
    - is_bmm1=True 时启用 L1 预取 (depthA1=1)
    """
    def align_up(x, alignment=16):
        return ((x + alignment - 1) // alignment) * alignment

    if single_m is None:
        single_m = m
    if single_n is None:
        single_n = n
    if single_k is None:
        single_k = k

    # 对于常看到的 flash attention 参数
    # bmm1: Q[Br,D] × K^T[D,Bc] → S[Br,Bc]: M=Br, N=Bc, K=D, B transposed
    # bmm2: P[Br,Bc] × V[Bc,D] → PV[Br,D]: M=Br, N=D, K=Bc
    # bmm1 启用 L1 预取: Q block 在多个 KV block 间复用，depthA1=1 让 Cube
    # 提前从 L1 加载下一个小分块到 L0A，减少 L0A 等待
    depth_a1 = 1 if is_bmm1 else 0
    depth_b1 = 0
    fields = {
        'usedCoreNum': used_cores,
        'M': m,
        'N': n,
        'Ka': k,       # K for matrix A input
        'Kb': k,       # K for matrix B input
        'singleCoreM': single_m,
        'singleCoreN': single_n,
        'singleCoreK': single_k,
        'baseM': base_m,
        'baseN': base_n,
        'baseK': base_k,
        'depthA1': depth_a1, 'depthB1': depth_b1,
        'stepM': 0, 'stepN': 0,
        'isBias': 0, 'transLength': 0,
        'iterateOrder': 0,
        'shareMode': 0,
        'shareL1Size': 0, 'shareL0CSize': 0, 'shareUbSize': 0,
        'batchM': 0, 'batchN': 0,
        'singleBatchM': 0, 'singleBatchN': 0,
        'stepKa': 0, 'stepKb': 0,
        'depthAL1CacheUB': 0, 'depthBL1CacheUB': 0,
        'dbL0A': 0, 'dbL0B': 0, 'dbL0C': 0,
        'ALayoutInfoB': 0, 'ALayoutInfoS': 0, 'ALayoutInfoN': 0,
        'ALayoutInfoG': 0, 'ALayoutInfoD': 0,
        'BLayoutInfoB': 0, 'BLayoutInfoS': 0, 'BLayoutInfoN': 0,
        'BLayoutInfoG': 0, 'BLayoutInfoD': 0,
        'CLayoutInfoB': 0,
        'CLayoutInfoS1': 0, 'CLayoutInfoN': 0, 'CLayoutInfoG': 0,
        'CLayoutInfoS2': 0,
        'BatchNum': 0, 'mxTypePara': 0,
    }

    # 50 int32 fields
    struct_fields = [
        fields['usedCoreNum'], fields['M'], fields['N'],
        fields['Ka'], fields['Kb'],
        fields['singleCoreM'], fields['singleCoreN'], fields['singleCoreK'],
        fields['baseM'], fields['baseN'], fields['baseK'],
        fields['depthA1'], fields['depthB1'],
        fields['stepM'], fields['stepN'],
        fields['isBias'], fields['transLength'], fields['iterateOrder'],
        fields['shareMode'],
        fields['shareL1Size'], fields['shareL0CSize'], fields['shareUbSize'],
        fields['batchM'], fields['batchN'],
        fields['singleBatchM'], fields['singleBatchN'],
        fields['stepKa'], fields['stepKb'],
        fields['depthAL1CacheUB'], fields['depthBL1CacheUB'],
        fields['dbL0A'], fields['dbL0B'], fields['dbL0C'],
        fields['ALayoutInfoB'], fields['ALayoutInfoS'], fields['ALayoutInfoN'],
        fields['ALayoutInfoG'], fields['ALayoutInfoD'],
        fields['BLayoutInfoB'], fields['BLayoutInfoS'], fields['BLayoutInfoN'],
        fields['BLayoutInfoG'], fields['BLayoutInfoD'],
        fields['CLayoutInfoB'],
        fields['CLayoutInfoS1'], fields['CLayoutInfoN'], fields['CLayoutInfoG'],
        fields['CLayoutInfoS2'],
        fields['BatchNum'], fields['mxTypePara'],
    ]

    return struct.pack('=50i', *struct_fields)


def generate_tiling(args):
    """生成完整的 tiling buffer"""
    td = TilingData(
        batch=args.batch,
        num_heads=args.heads_q,
        num_kv_heads=args.heads_kv,
        seq_q=args.seq_q,
        seq_k=args.seq_k,
        head_dim=args.dim,
        q_block_size=args.br,
        kv_block_size=args.bc,
        is_causal=args.causal,
        block_split_mode=args.split_2d,
    )

    # bmm1: Q[Br,D] × K^T[D,Bc] → S[Br,Bc]
    # 注意: Matmul B 是 K^T (transposed)
    # 使用自适应 base: D≥128 时 base_m=32 提高 Cube 效率，否则 16
    b1_base_m = 32 if args.dim >= 128 else 16
    tiling_qk = make_tcube_tiling(
        m=args.br, n=args.bc, k=args.dim,
        base_m=b1_base_m, base_n=16, base_k=16,
        used_cores=1, is_bmm1=True,
    )

    # bmm2: P[Br,Bc] × V[Bc,D] → PV[Br,D]
    tiling_pv = make_tcube_tiling(
        m=args.br, n=args.dim, k=args.bc,
        base_m=16, base_n=16, base_k=16,
        used_cores=1, is_bmm1=False,
    )

    # 拼接: FlashAttentionTilingData + TCubeTiling(bmm1) + TCubeTiling(bmm2)
    buffer = td.pack() + tiling_qk + tiling_pv

    return buffer, td


def main():
    parser = argparse.ArgumentParser(description="FlashAttention Tiling 数据生成器")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--seq-q", type=int, default=2048)
    parser.add_argument("--seq-k", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--br", type=int, default=64, help="Q block size")
    parser.add_argument("--bc", type=int, default=64, help="KV block size")
    parser.add_argument("--causal", type=int, default=1, help="Enable causal mask")
    parser.add_argument("--split-2d", type=int, default=0, help="2D core split")
    parser.add_argument("--out", type=str, default="", help="Output .bin file")
    parser.add_argument("--print", action="store_true", help="Print hex dump")
    parser.add_argument("--verify", action="store_true", help="Verify workspace size")
    args = parser.parse_args()

    buffer, td = generate_tiling(args)
    size = len(buffer)

    print(f"Tiling buffer: {size} bytes", file=sys.stderr)
    print(f"  - FlashAttentionTilingData: {TILING_DATA_SIZE} bytes", file=sys.stderr)
    print(f"  - TCubeTiling(bmm1): {TCUBE_TILING_SIZE} bytes @ offset {td.tiling_qk_offset}", file=sys.stderr)
    print(f"  - TCubeTiling(bmm2): {TCUBE_TILING_SIZE} bytes @ offset {td.tiling_pv_offset}", file=sys.stderr)
    print(f"  - Workspace: {td.workspace_size} bytes ({td.workspace_size/1024:.1f} KB)", file=sys.stderr)
    print(f"\n{td}", file=sys.stderr)

    if args.verify:
        # 验证 workspace 大小合理性
        B, Hq, Hkv, Sq, Sk, D = args.batch, args.heads_q, args.heads_kv, args.seq_q, args.seq_k, args.dim
        Br, Bc = args.br, args.bc
        total_q = B * Hq * Sq * D * 2  # half=2B
        total_kv = B * Hkv * Sk * D * 2 * 2  # K+V
        workspace = td.workspace_size
        print(f"\n  Verification:", file=sys.stderr)
        print(f"    Q  size: {total_q/1024:.1f} KB", file=sys.stderr)
        print(f"    KV size: {total_kv/1024:.1f} KB", file=sys.stderr)
        print(f"    W  size: {workspace/1024:.1f} KB", file=sys.stderr)
        max_ws = total_q + total_kv
        ratio = workspace / max_ws * 100 if max_ws > 0 else 0
        print(f"    W / (Q+KV) = {ratio:.1f}% (reasonable < 30% for flash attn)", file=sys.stderr)

    if args.out:
        with open(args.out, "wb") as f:
            f.write(buffer)
        print(f"\n写入 {args.out} ({size} bytes)", file=sys.stderr)
    else:
        # 输出到 stdout（二进制）
        sys.stdout.buffer.write(buffer)

    if args.print:
        print("\nHex dump (first 128 bytes):", file=sys.stderr)
        for i in range(0, min(128, size), 16):
            chunk = buffer[i:i+16]
            hex_str = " ".join(f"{b:02x}" for b in chunk)
            ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            print(f"  {i:04x}: {hex_str:48s} {ascii_str}", file=sys.stderr)


if __name__ == "__main__":
    main()

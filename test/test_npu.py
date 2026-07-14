#!/usr/bin/env python3
"""
FlashAttention NPU 核函数测试
============================
在 NPU 上加载并运行 flash_attention_main 核函数，验证正确性。

用法:
  python3 test_npu.py                                          # 默认参数
  python3 test_npu.py --heads-q 4  --heads-kv 1 --dim 64       # 小规模
  python3 test_npu.py --heads-q 32 --heads-kv 8 --dim 128      # 标准 GQA

依赖:
  - CANN 9.0.0 (ASCEND_HOME_PATH)
  - NPU 驱动 (npu-smi 可用)
  - numpy, (可选 torch)
"""

import argparse
import math
import os
import sys
import struct
import numpy as np

NPU_AVAILABLE = False
try:
    import acl
    NPU_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from gen_tiling import generate_tiling, TILING_DATA_SIZE, TCUBE_TILING_SIZE


# ============================================================================
# NPU 运行时 (通过 AscendCL)
# ============================================================================

class NPURuntime:
    """AscendCL 运行时封装 — 管理设备、上下文、内存"""

    def __init__(self):
        if not NPU_AVAILABLE:
            raise RuntimeError("acl 模块不可用。请在 NPU 环境下运行。")
        self.initialized = False
        self.device_id = 0
        self.context = None
        self.stream = None

    def init(self):
        ret = acl.init()
        assert ret == 0, f"acl.init failed: {ret}"
        ret = acl.rt.set_device(self.device_id)
        assert ret == 0, f"acl.rt.set_device failed: {ret}"
        self.context, ret = acl.rt.create_context(self.device_id)
        assert ret == 0, f"acl.rt.create_context failed: {ret}"
        self.stream, ret = acl.rt.create_stream()
        assert ret == 0, f"acl.rt.create_stream failed: {ret}"
        self.initialized = True
        print(f"[NPU] 初始化成功: device={self.device_id}")

    def malloc(self, size):
        """分配 NPU 设备内存"""
        buf, ret = acl.rt.malloc(size, acl.constants.ACL_MEM_MALLOC_HUGE_FIRST)
        assert ret == 0, f"acl.rt.malloc({size}) failed: {ret}"
        return buf

    def memcpy_to_device(self, buf, data):
        """将 numpy 数据拷贝到 NPU 设备内存"""
        nbytes = data.nbytes
        ret = acl.rt.memcpy(buf, nbytes, data.ctypes.data, nbytes,
                            acl.constants.ACL_MEMCPY_HOST_TO_DEVICE)
        assert ret == 0, f"acl.rt.memcpy H2D failed: {ret}"

    def memcpy_to_host(self, data, buf, nbytes):
        """从 NPU 设备内存拷贝到 numpy 数组"""
        ret = acl.rt.memcpy(data.ctypes.data, nbytes, buf, nbytes,
                            acl.constants.ACL_MEMCPY_DEVICE_TO_HOST)
        assert ret == 0, f"acl.rt.memcpy D2H failed: {ret}"

    def load_kernel(self, kernel_path, kernel_name="flash_attention_main"):
        """加载核函数 .o 文件"""
        with open(kernel_path, "rb") as f:
            kernel_data = f.read()
        # CANN 8.0+/9.0.0: acl.rt.load_command_from_file
        # 如果此 API 不可用，尝试 acl.rt.load_command_from_mem(data, len)
        try:
            self.kernel_handle, ret = acl.rt.load_command_from_file(kernel_path)
            if ret != 0:
                raise RuntimeError(f"load_command_from_file ret={ret}")
        except AttributeError:
            # Fallback: load from memory
            self.kernel_handle, ret = acl.rt.load_command_from_mem(kernel_data, len(kernel_data))
            if ret != 0:
                raise RuntimeError(f"load_command_from_mem ret={ret}")
        print(f"[NPU] 加载 kernel: {kernel_path} ({len(kernel_data)} bytes)")

    def launch_kernel(self, args_list, block_dim=1):
        """启动核函数
        args_list: [(buf, size), ...] 实际仅使用 buf (设备内存指针)
        """
        # 提取设备内存指针 (kernel 参数: q, k, v, o, w, t)
        arg_ptrs = [buf for buf, _ in args_list]
        # CANN 8.0+/9.0.0: acl.rt.launch_command(handle, blockDim, args, stream)
        ret = acl.rt.launch_command(self.kernel_handle, block_dim, arg_ptrs, self.stream)
        if ret != 0:
            raise RuntimeError(f"launch_command failed: {ret}")
        # 等待核函数执行完成
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != 0:
            raise RuntimeError(f"synchronize_stream failed: {ret}")
        print(f"[NPU] 核函数执行完成 (blockDim={block_dim})")

    def destroy(self):
        if self.initialized:
            try:
                if hasattr(self, 'kernel_handle'):
                    acl.rt.unload_command(self.kernel_handle)
            except Exception:
                pass
            if self.stream:
                acl.rt.destroy_stream(self.stream)
            if self.context:
                acl.rt.destroy_context(self.context)
            acl.rt.reset_device(self.device_id)
            acl.finalize()
            self.initialized = False
            print("[NPU] 已释放资源")


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
# 测试主逻辑
# ============================================================================

def run_test(args):
    B, Hq, Hkv, Sq, Sk, D = args.batch, args.heads_q, args.heads_kv, args.seq_q, args.seq_k, args.dim
    causal = bool(args.causal)

    print("=" * 60)
    print("FlashAttention NPU 核函数测试")
    print("=" * 60)
    print(f"\n[配置] B={B} Hq={Hq} Hkv={Hkv} Sq={Sq} Sk={Sk} D={D}")
    print(f"       GQA: {Hq//Hkv}x, causal={causal}")
    print(f"       总 Q: {B*Hq*Sq*D:,} 元素 ({B*Hq*Sq*D*2/1024/1024:.1f} MB)")
    print(f"       总 KV: {B*Hkv*Sk*D*2:,} 元素 ({B*Hkv*Sk*D*2*2/1024/1024:.1f} MB)")

    # ── 1. 生成 tiling 数据 ──
    class FakeArgs: pass
    ta = FakeArgs()
    for k in ['batch','heads_q','heads_kv','seq_q','seq_k','dim','br','bc','causal','split_2d']:
        setattr(ta, k, getattr(args, k, None))
    ta.br = args.br if args.br else min(64, args.dim)
    ta.bc = args.bc if args.bc else 64

    tiling_buf, td = generate_tiling(ta)
    workspace_bytes = td.workspace_size

    # tiling 数据总大小 (FlashAttentionTilingData + 2×TCubeTiling)
    tiling_total = len(tiling_buf)
    print(f"\n[Tiling] {tiling_total} bytes")
    print(f"   workspace 需求: {workspace_bytes} bytes ({workspace_bytes/1024:.1f} KB)")

    # ── 2. 生成随机输入 ──
    np.random.seed(42)
    q_np = np.random.randn(B, Hq, Sq, D).astype(np.float16)
    k_np = np.random.randn(B, Hkv, Sk, D).astype(np.float16)
    v_np = np.random.randn(B, Hkv, Sk, D).astype(np.float16)
    print(f"\n[输入] Q: {q_np.shape} {q_np.dtype}")
    print(f"       K: {k_np.shape} {k_np.dtype}")
    print(f"       V: {v_np.shape} {v_np.dtype}")

    # ── 3. 计算参考输出 ──
    print(f"\n[参考] 计算参考输出...")
    if torch is not None:
        q_t = torch.from_numpy(q_np.copy())
        k_t = torch.from_numpy(k_np.copy())
        v_t = torch.from_numpy(v_np.copy())
        ref_out = flash_attention_ref(q_t, k_t, v_t, causal=causal).numpy()
    else:
        ref_out = flash_attention_ref_np(q_np, k_np, v_np, causal=causal)
    print(f"       形状: {ref_out.shape}, 范围: [{ref_out.min():.4f}, {ref_out.max():.4f}]")

    # ── 4. 保存参考输出到文件 (供 NPU 对比) ──
    ref_path = "ref_out.bin"
    ref_out.tofile(ref_path)
    print(f"       已保存: {ref_path} ({ref_out.nbytes} bytes)")

    # ── 5. 检查 NPU 环境并运行核函数 ──
    out_npu = None
    if not NPU_AVAILABLE:
        print(f"\n[NPU] ⚡ acl 模块不可用，跳过核函数执行")
        print(f"[NPU] 请将本脚本拷贝到 NPU 机器上运行")
    else:
        try:
            runtime = NPURuntime()
            runtime.init()

            # 查找 kernel .o 文件
            kernel_path = args.kernel
            if not kernel_path or not os.path.exists(kernel_path):
                default = "build/CMakeFiles/flash_attention_kernel.dir/op_kernel/flash_attention.asc.o"
                if os.path.exists(default):
                    kernel_path = default
            if not kernel_path or not os.path.exists(kernel_path):
                raise FileNotFoundError(f"找不到 kernel .o 文件")

            # 检查文件类型
            import subprocess
            file_info = subprocess.run(
                ["file", "-b", kernel_path], capture_output=True, text=True
            ).stdout.strip()
            print(f"\n[NPU] Kernel: {kernel_path}")
            print(f"       类型: {file_info}")
            if "aarch64" not in file_info and "ELF" not in file_info:
                print(f"       ⚠ 可能不是 NPU 目标")

            # 分配设备内存
            q_size = q_np.nbytes
            k_size = k_np.nbytes
            v_size = v_np.nbytes
            o_size = B * Hq * Sq * D * np.dtype(np.float16).itemsize
            w_size = workspace_bytes
            t_size = tiling_total

            d_q = runtime.malloc(q_size)
            d_k = runtime.malloc(k_size)
            d_v = runtime.malloc(v_size)
            d_o = runtime.malloc(o_size)
            d_w = runtime.malloc(w_size)
            d_t = runtime.malloc(t_size)

            # 拷贝输入数据到 NPU
            runtime.memcpy_to_device(d_q, q_np)
            runtime.memcpy_to_device(d_k, k_np)
            runtime.memcpy_to_device(d_v, v_np)
            runtime.memcpy_to_device(d_t, np.frombuffer(tiling_buf, dtype=np.uint8))

            print(f"\n[NPU] 设备内存分配:")
            print(f"       Q: {q_size/1024:.1f} KB")
            print(f"       K: {k_size/1024:.1f} KB")
            print(f"       V: {v_size/1024:.1f} KB")
            print(f"       O: {o_size/1024:.1f} KB (输出)")
            print(f"       W: {w_size/1024:.1f} KB (workspace)")
            print(f"       T: {t_size} bytes (tiling)")

            # 加载核函数
            runtime.load_kernel(kernel_path)

            # ⚡ 执行核函数
            # flash_attention_main(q, k, v, o, w, t)
            args_list = [
                (d_q, q_size), (d_k, k_size), (d_v, v_size),
                (d_o, o_size), (d_w, w_size), (d_t, t_size),
            ]
            runtime.launch_kernel(args_list, block_dim=8)

            # 读取输出
            out_npu = np.empty(B * Hq * Sq * D, dtype=np.float16)
            runtime.memcpy_to_host(out_npu, d_o, o_size)
            out_npu = out_npu.reshape(B, Hq, Sq, D)

            # 保存 NPU 输出
            npu_path = "npu_out.bin"
            out_npu.tofile(npu_path)
            print(f"\n[NPU] 输出已保存: {npu_path}")

            runtime.destroy()

        except Exception as e:
            print(f"\n[NPU] ❌ 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 6. 结果对比 ──
    print(f"\n{'='*60}")
    if out_npu is not None:
        ref_f = ref_out.astype(np.float32)
        npu_f = out_npu.astype(np.float32)
        cos_sim = np.dot(ref_f.flatten(), npu_f.flatten()) / (
            np.linalg.norm(ref_f) * np.linalg.norm(npu_f) + 1e-10)
        max_diff = np.max(np.abs(ref_f - npu_f))
        mean_diff = np.mean(np.abs(ref_f - npu_f))

        print(f"[结果] NPU 核函数执行完成")
        print(f"       参考输出: [{ref_out.min():.4f}, {ref_out.max():.4f}]")
        print(f"       NPU 输出: [{out_npu.min():.4f}, {out_npu.max():.4f}]")
        print(f"       cos_sim  = {cos_sim:.6f}  (应 > 0.99)")
        print(f"       max_diff = {max_diff:.6f}  (应 < 0.1)")
        print(f"       mean_diff= {mean_diff:.6f}  (应 < 0.01)")

        if cos_sim > 0.99:
            print(f"\n✅ 测试通过! 核函数输出与参考一致")
        elif cos_sim > 0.9:
            print(f"\n⚠️ 精度略差，可能存在数值误差累积")
        else:
            print(f"\n❌ 测试失败! 核函数输出与参考偏差过大")
    else:
        print(f"[结果] 参考输出已生成 ({ref_out.shape})")
        print(f"       将本目录拷贝到 NPU 机器后运行 test_npu.py")
    print(f"{'='*60}")

    return out_npu, ref_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention NPU 测试")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=4)
    parser.add_argument("--heads-kv", type=int, default=1)
    parser.add_argument("--seq-q", type=int, default=256)
    parser.add_argument("--seq-k", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--br", type=int, default=0, help="Q tile size")
    parser.add_argument("--bc", type=int, default=0, help="KV tile size")
    parser.add_argument("--causal", type=int, default=1)
    parser.add_argument("--split-2d", type=int, default=0)
    parser.add_argument("--kernel", type=str, default=None, help="Path to .asc.o")
    args = parser.parse_args()

    run_test(args)

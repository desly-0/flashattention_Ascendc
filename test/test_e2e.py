#!/usr/bin/env python3
"""
FlashAttention V2 — 端到端测试
==================================
验证完整流程: tiling 生成 → 核函数构建 → 正确性验证

用法:
  python3 test_e2e.py                   # 默认参数 (无 NPU 仿真)
  python3 test_e2e.py --npu             # NPU 模式 (需硬件)
"""

import argparse
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

try:
    import numpy as np
except ImportError:
    print("需要 numpy: pip install numpy")
    sys.exit(1)


def check_cann_env():
    """检查 CANN 环境是否配置"""
    ascend_home = os.environ.get('ASCEND_HOME_PATH', '')
    if not ascend_home:
        candidates = [
            '/home/hello/Ascend/cann-8.5.0',
            '/usr/local/Ascend/cann-8.5.0',
            '/usr/local/Ascend/cann-9.0.0',
        ]
        for c in candidates:
            if os.path.isdir(c):
                ascend_home = c
                break
    return ascend_home


def run_cmd(cmd, desc=""):
    """运行命令并打印输出"""
    if desc:
        print(f"\n[{desc}]")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split('\n')[-20:]:
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  [ERROR] return code: {result.returncode}")
        if result.stderr:
            stderr_lines = result.stderr.strip().split('\n')[-10:]
            for line in stderr_lines:
                print(f"  [STDERR] {line}")
    return result.returncode == 0


def run_test(args):
    print("=" * 60)
    print("FlashAttention V2 端到端测试")
    print("=" * 60)

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tools_dir = os.path.join(project_dir, 'tools')
    test_dir = os.path.join(project_dir, 'test')

    ascend_home = check_cann_env()
    print(f"\n[环境]")
    print(f"  项目目录: {project_dir}")
    print(f"  ASCEND_HOME_PATH: {ascend_home or '(未设置)'}")

    B, Hq, Hkv, Sq, Sk, D = args.batch, args.heads_q, args.heads_kv, args.seq_q, args.seq_k, args.dim
    causal = args.causal
    Br = args.br if args.br else min(64, D)
    Bc = args.bc if args.bc else 128

    print(f"  测试参数: B={B} Hq={Hq} Hkv={Hkv} Sq={Sq} Sk={Sk} D={D}")
    print(f"  causal={causal} Br={Br} Bc={Bc}")

    # ── 步骤 1: 测试 tiling 生成 ──
    print(f"\n{'─'*60}")
    print("步骤 1: Tiling 生成")
    sys.path.insert(0, tools_dir)
    from gen_tiling import generate_tiling

    class FakeArgs: pass
    ta = FakeArgs()
    for k in ['batch','heads_q','heads_kv','seq_q','seq_k','dim','br','bc','causal','split_2d']:
        setattr(ta, k, getattr(args, k, None))
    ta.br = Br; ta.bc = Bc; ta.split_2d = getattr(args, 'split_2d', 0)
    ta.causal = causal

    tiling_buf, td = generate_tiling(ta)
    print(f"  ✓ Tiling 数据: {len(tiling_buf)} bytes")
    print(f"  ✓ Workspace: {td['workspaceSize']} bytes ({td['workspaceSize']/1024:.1f} KB)")
    if td['scale'] - 1.0/math.sqrt(D) < 1e-6:
        print(f"  ✓ Scale = {td['scale']:.6f}")
    else:
        print(f"  ✗ Scale 错误!")

    # ── 步骤 2: 运行参考实现 ──
    print(f"\n{'─'*60}")
    print("步骤 2: 参考实现测试")
    ref_script = os.path.join(test_dir, 'test_ref.py')
    if os.path.exists(ref_script):
        cmd = [
            sys.executable, ref_script,
            '--batch', str(B), '--heads-q', str(Hq), '--heads-kv', str(Hkv),
            '--seq-q', str(Sq), '--seq-k', str(Sk), '--dim', str(D),
            '--br', str(Br), '--bc', str(Bc),
        ]
        if causal:
            cmd.append('--causal')
        ok = run_cmd(cmd, "test_ref.py")
        if not ok:
            print("  ⚠ 参考实现测试不完整")

    # ── 步骤 3: 构建核函数 (CPU 仿真) ──
    print(f"\n{'─'*60}")
    print("步骤 3: 核函数构建")

    build_script = os.path.join(tools_dir, 'build.sh')
    if os.path.exists(build_script):
        # 先清理再构建 sim 模式
        run_cmd(['bash', build_script, 'clean'])
        ok = run_cmd(['bash', build_script, 'sim'], "build.sh sim")
        if ok:
            print("  ✓ 核函数构建成功")
        else:
            print("  ✗ 核函数构建失败")
    else:
        print(f"  ⚠ 找不到 build.sh")

    # ── 步骤 4: 检查编译产物 ──
    print(f"\n{'─'*60}")
    print("步骤 4: 编译产物检查")
    build_dir = os.path.join(tools_dir, 'build')
    o_files = []
    if os.path.isdir(build_dir):
        for root, dirs, files in os.walk(build_dir):
            for f in files:
                if f.endswith('.o'):
                    o_files.append(os.path.join(root, f))

    if o_files:
        for of in o_files:
            size_kb = os.path.getsize(of) / 1024
            file_type = subprocess.run(['file', '-b', of],
                                       capture_output=True, text=True).stdout.strip()
            print(f"  ✓ {os.path.relpath(of, project_dir)} ({size_kb:.1f} KB)")
            print(f"    类型: {file_type}")
    else:
        print(f"  ⚠ 未找到编译产物")

    # ── 步骤 5: NPU 测试 (如果启用) ──
    if args.npu:
        print(f"\n{'─'*60}")
        print("步骤 5: NPU 核函数测试")
        npu_script = os.path.join(test_dir, 'test_npu.py')
        if os.path.exists(npu_script):
            cmd = [
                sys.executable, npu_script,
                '--batch', str(B), '--heads-q', str(Hq), '--heads-kv', str(Hkv),
                '--seq-q', str(Sq), '--seq-k', str(Sk), '--dim', str(D),
                '--br', str(Br), '--bc', str(Bc),
                '--causal', str(int(causal)),
            ]
            if o_files:
                cmd.extend(['--kernel', o_files[0]])
            run_cmd(cmd, "test_npu.py")
        else:
            print(f"  ⚠ 找不到 test_npu.py")
    else:
        print(f"\n{'─'*60}")
        print("步骤 5: NPU 测试 (跳过, 用 --npu 启用)")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("端到端测试完成")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttention V2 端到端测试")
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
    parser.add_argument("--npu", action="store_true", help="Enable NPU test")
    args = parser.parse_args()

    run_test(args)

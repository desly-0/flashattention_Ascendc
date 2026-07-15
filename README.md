# FlashAttention Ascend C — V1

Ascend 910B NPU 上的 FlashAttention 算子实现，使用 Ascend C 开发。

## 原理

FlashAttention 通过分块（tiling）和 online softmax 合并，将 QK^T 注意力矩阵的计算与 softmax + PV 累加重叠，避免显存中物化完整的 S 矩阵（Sq×Sk），使得计算复杂度从 O(n²) 降低为接近 O(n)。

**核心流水线：**

```
KV Block 循环:
  bi=0:  ┌─ bmm1[0] (Q×K^T, async)
         └→ wait → softmax(S[0])  ‖  bmm1[1] (async, overlap)
                     → Cast P    → bmm2[0] (P×V, async)  ‖  rescale (vector)
                                    → wait → O += rescale*O + inv_l*PV
  bi=1:  wait bmm1[1] → start bmm1[2] → online-merge softmax(S[1]) → ...
```

**online softmax 合并：**
- 每个 KV block 算出局部 softmax 的 m（最大值）和 l（exp 和）
- 合并时对之前累积的 O 做 rescale：`O = exp(m_old - m_new) * l_old / l_new * O + P * V / l_new`
- 避免了完整的 S 矩阵在显存中物化

**多核拆分：** 支持按 head 维度 + 按 seq_len 维度二维切分

## 实现内容

- 核函数 (`flash_attention.asc`)：bmm1 + online-merge softmax + bmm2 + O 累加 四阶段流水线
- 向量化路径：m/l 初始化（Duplicate）、rescale 因子计算（Sub+Exp+Mul+Recip）、m/l 保存（DataCopy）全部向量化
- Cube 异步启动 + L1 预取（bmm1 depthA1=1），与 AIV 重叠执行
- 自适应 base_m（D≥128 时 32，否则 16）
- 测试：Python 参考实现（PyTorch/numpy）+ NPU 核函数测试
- 基准测试：支持 NPU kernel 启动耗时和 PyTorch SDPA 基线对比

## 构建

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash tools/build.sh npu      # NPU 编译
```

## 测试

```bash
# 参考正确性（无需 NPU）
python3 test/test_ref.py --heads-q 32 --heads-kv 8 --seq-q 2048 --seq-k 2048 --dim 128

# NPU 核函数验证
python3 test/test_npu.py --heads-q 4 --heads-kv 1 --seq-q 256 --seq-k 256 --dim 64

# 性能
python3 bench/bench_npu.py --heads-q 32 --heads-kv 8 --seq-q 2048 --seq-k 2048 --dim 128
```

## 目录

```
V1/
├── src/
│   ├── flash_attention.asc      # 核函数
│   ├── flash_attention_tiling.h # Tiling 数据结构
│   ├── flash_attention.cpp      # Tiling 入口
│   └── op_host/                 # Host 注册
├── test/
│   ├── test_npu.py              # NPU 测试（AscendCL）
│   └── test_ref.py              # 参考实现
├── bench/
│   └── bench_npu.py             # 性能基准
├── tools/
│   ├── gen_tiling.py            # Tiling 数据生成
│   ├── build.sh                 # 编译脚本
│   └── CMakeLists.txt
└── .claude/skills/ascend-c-dev/ # Ascend C 开发 skill
```

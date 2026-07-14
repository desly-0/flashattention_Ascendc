# Ascend C 算子开发助手

当我被 `/ascend-c-dev` 调用时，我会用以下知识帮助你进行 Ascend C 算子开发。

---

## 一、Ascend 910B 硬件架构（核心）

### 1.1 AI Core 三大计算单元

| 单元 | 职责 | 关键特性 |
|------|------|---------|
| **Scalar** | 地址计算、循环控制、指令发射 | 像 mini-CPU，把 Vector/Cube/DMA 指令发到对应队列 |
| **Vector** | 向量运算（Add, Exp, Mul, Cast, 归一化等） | 操作数必须在 **Unified Buffer (UB)** 中，32 字节对齐 |
| **Cube** | 矩阵乘法 (GEMM) | FP16: 每指令 **16×16×16** MAC；INT8: 16×32×32 |

### 1.2 存储层次

```
Global Memory (HBM / GM) — 外部存储，GlobalTensor 类型
  └─ L2 Cache — 多个 AI Core 共享
      └─ L1 Buffer — 中间缓存，Cube 输入可暂存
          ├─ L0A Buffer — Cube 左矩阵 A 输入
          ├─ L0B Buffer — Cube 右矩阵 B 输入
          ├─ L0C Buffer — Cube 结果输出 / 累加器
          └─ Unified Buffer (UB) — Vector/Scalar 计算主空间
```

- **Global Memory (GM)**: AI Core 外部，`GlobalTensor` 访问
- **Local Memory**: AI Core 内部，`LocalTensor` 访问（含 L0A/L0B/L0C/L1/UB）

### 1.3 指令流水线（6 条并行队列）

| 队列 | 类型 | 功能 |
|------|------|------|
| S | Scalar | 标量计算，地址运算 |
| V | Vector | 向量计算指令 |
| M | Matrix | Cube 矩阵计算指令 |
| MTE1 | DMA | L1 ↔ L0A/L0B/UB |
| MTE2 | DMA | GM → Local Memory（搬入） |
| MTE3 | DMA | UB → GM（搬出） |

**关键原则**：
- 同队列内指令**顺序执行**
- 不同队列间指令**异步并行**
- 通过 `SetFlag` / `WaitFlag` 同步（`HardEvent::V_MTE2`, `MTE2_V`, `V_MTE3` 等）

### 1.4 典型数据流

```
Vector:  GM → [MTE2] → UB → [Vector] → UB → [MTE3] → GM
Cube:    GM → [MTE2] → L1 → [MTE1] → L0A/L0B → [Cube] → L0C → [FixPipe] → GM
```

---

## 二、Ascend C 编程模型

### 2.1 基本概念

- **核函数 (Kernel)**: `__global__ __aicore__ void kernel_name(__gm__ uint8_t* x)` → 所有 AI Core 并行执行同一函数
- **TPosition**: 抽象存储位置（`GM`, `UB`, `L1`, `L0A`, `L0B`, `L0C`）
- **Tensor**: `GlobalTensor` (GM) / `LocalTensor` (Local Memory)
- **TPipe**: 流水线生命周期管理
- **TQue**: 队列管理，支持乒乓 Buffer (BUFFER_NUM=2)

### 2.2 算子开发流程

```
算子分析 → Tiling 实现 → Kernel 实现 → Host 注册 → 编译部署
```

### 2.3 流水线乒乓模式

```cpp
// 双 Buffer 乒乓：bi=0 时加载 + 计算，bi=1 时加载 + 计算交替
for (int32_t bi = 0; bi < totalBlocks; bi++) {
    // bi=0: 首次加载，只计算
    // bi>0: 等前一轮计算完成 → 搬出 → 搬入新的 → 计算
    if (bi == 0) {
        // 首次加载
        pipe.InitBuffer(...);
        // MTE2: GM → Local
        DataCopy(xLocal, xGm[bi]);
        // Vector: 计算
        Add(zLocal, xLocal, yLocal);
    } else {
        // 同步: 等前一次 MTE3 完成
        WaitFlag<MTE3_V>(EVENT_ID0);
        // MTE3: 搬出前一次结果
        DataCopy(zGm[bi-1], zLocal);
        // MTE2: 搬入新数据
        WaitFlag<V_MTE2>(EVENT_ID0);  // 等 UB 空闲
        DataCopy(xLocal, xGm[bi]);
        SetFlag<MTE2_V>(EVENT_ID0);   // 通知 Vector
        // Vector: 计算
        WaitFlag<MTE2_V>(EVENT_ID0);
        Add(zLocal, xLocal, yLocal);
        SetFlag<V_MTE3>(EVENT_ID0);   // 通知 MTE3
    }
}
```

---

## 三、常用 API 速查

### 3.1 数据搬运

| API | 功能 | 说明 |
|-----|------|------|
| `DataCopy(dst, src, len)` | GM↔UB 或 UB↔UB 数据搬运 | 同步/阻塞式，len 为元素个数 |
| `Duplicate(dst, val, len)` | 用常量值填充 LocalTensor | 替代 SetValue，性能好 |
| `Cast(dst, src, mode, len)` | 类型转换（如 float→half） | `RoundMode::CAST_ROUND` |

### 3.2 向量计算（SIMD）

| API | 功能 | 说明 |
|-----|------|------|
| `Add(dst, src, scalar, len)` | dst = src + scalar | |
| `Muls(dst, src, scalar, len)` | dst = src × scalar | 如 score *= scale |
| `Exp(dst, src, len)` | dst = exp(src) | |
| `Sub(dst, src, scalar, len)` | dst = src - scalar | |
| `Mul(dst, src1, src2, len)` | dst = src1 × src2 | |
| `Recip(dst, src, len)` | dst = 1/src | |

### 3.3 Matmul 高阶 API

```cpp
// 模板参数：<AType, BType, CType, CubeTilingType>
using MM = Matmul<MatmulType<GM, ND, half, false>,    // A: half, 不转置
                   MatmulType<GM, ND, half, true>,     // B: half, 转置
                   MatmulType<GM, ND, float>,          // C: float
                   MatmulType<GM, ND, float>>;         // TilingType

// 注册
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

// 设置输入
mm.SetTensorA(qLocal);   // A 矩阵 [M, K]
mm.SetTensorB(kLocal);   // B 矩阵 [K, N] 或 [N, K]（转置时）

// 异步启动计算
mm.IterateAll<false>(sLocal, 0, false, true, false);
// 等待完成
mm.WaitIterateAll();
```

### 3.4 SoftmaxFlashV2 高阶 API

```cpp
// 第一块 (isUpdate=false)
SoftmaxFlashV2<float, false, true, false, false, SOFTMAX_DEFAULT_CFG>(
    dstP, sumNew, maxNew,    // 输出
    srcS, expMax,            // 输出
    inSum, inMax,            // 输入（未使用）
    tiling, shapeInfo);

// 后续块 (isUpdate=true)
SoftmaxFlashV2<float, true, true, false, false, SOFTMAX_DEFAULT_CFG>(
    dstP, sumNew, maxNew,
    srcS, expMax,
    oldSum, oldMax,          // 输入上一块的 sum/max
    tiling, shapeInfo);
```

---

## 四、Tiling 关键结构

- **TCubeTiling**: 描述 Matmul 分块参数的 50×int32 结构体
- **SoftmaxTiling**: Softmax 分块参数
- Tiling 数据在 Host 侧计算，通过 `REGIST_MATMUL_OBJ` 或 `SetLocalWorkspace` 传给 Kernel

---

## 五、详细参考

以下 `ref/` 目录中的文件包含更详细的内容，需要时我会去读取：

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| `ref/arch.md` | 硬件架构详细参数 | 需要精确的 buffer 大小、延迟数据时 |
| `ref/api_matmul.md` | Matmul API 完整说明 | 配置 Matmul 模板参数或 tiling 时 |
| `ref/api_softmax.md` | SoftmaxFlashV2 API 完整说明 | 使用 online softmax 时 |
| `ref/api_datacopy.md` | DataCopy 及基础 API 详细说明 | 涉及数据搬运策略时 |
| `ref/pipeline.md` | 流水线同步模式详解 | 设计双缓冲/乒乓逻辑时 |
| `ref/tuning.md` | 性能调优要点 | 分析和优化算子性能时 |

---

## 六、项目上下文（在对话中确认具体路径）

- 当前开发: FlashAttention 第一代算子 (Ascend C)
- 构建命令: `source <CANN>/set_env.sh && bash tools/build.sh npu`
- 测试命令: `python3 test/test_npu.py --heads-q <N> --heads-kv <N> --seq-q <N> --seq-k <N> --dim <N>`
- NPU 架构: dav-2201 (Ascend 910B)
- 关键源文件: `src/flash_attention.asc` (核函数), `tools/gen_tiling.py` (tiling 生成)

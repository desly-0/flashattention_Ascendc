# Ascend 910B 硬件架构详细参考

## AI Core 内部组件

| 组件 | 功能 | 备注 |
|------|------|------|
| **Scalar** | 地址计算、循环控制、分支跳转；发射 Vector/Cube/MTE 指令 | 有自己的 iCache (16-32KB) 和 dCache (16KB) |
| **Vector** | 向量运算：Add/Mul/Exp/Cast/Reduce 等 | 数据必须在 UB 中，32 字节对齐 |
| **Cube** | 矩阵乘法：FP16 16×16×16 / INT8 16×32×32 | A 来自 L0A, B 来自 L0B, 结果到 L0C |
| **MTE1** | L1 ↔ L0A/L0B 搬运 | Cube 输入数据准备 |
| **MTE2** | GM → Local Memory (L1/L0A/L0B/UB) | 数据搬入 |
| **MTE3** | UB → GM | 数据搬出 |
| **FixPipe** | L0C → GM/L1，数据格式转换 | 反量化、类型转换 |

## 指令流水线

AI Core 有 6 条独立的指令队列并行执行：

- **S 队列**: Scalar 计算
- **V 队列**: Vector 计算
- **M 队列**: Matrix (Cube) 计算
- **MTE1 队列**: L1 与 L0 之间搬运
- **MTE2 队列**: GM 搬入到 Local Memory
- **MTE3 队列**: UB 搬出到 GM

### 同步事件 (SetFlag / WaitFlag)

```cpp
// Vector 等 MTE2 搬入完成
WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
// MTE2 通知 Vector 数据已就绪
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
// Vector 通知 MTE3 可以搬出结果
SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
// MTE3 通知 Vector 数据已搬出
WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
```

## 存储层次参考

| 存储 | 位置 | 数据局部性 | 访问方式 |
|------|------|-----------|---------|
| Global Memory (HBM) | 片外 | 所有 AI Core 共享 | GlobalTensor |
| L2 Cache | 片内 | 所有 AI Core 共享 | 自动缓存 |
| L1 Buffer | 片内 | 单 AI Core 私有 | LocalTensor |
| L0A Buffer | 片内 | 单 AI Core 私有 | Cube 左矩阵输入 |
| L0B Buffer | 片内 | 单 AI Core 私有 | Cube 右矩阵输入 |
| L0C Buffer | 片内 | 单 AI Core 私有 | Cube 输出/累加 |
| Unified Buffer (UB) | 片内 | 单 AI Core 私有 | Vector/Scalar 计算 |

## NPU 架构版本

- `dav-2201`: Ascend 910B1 / 910B2 / 910B3
- `dav-3510`: Ascend 950 系列

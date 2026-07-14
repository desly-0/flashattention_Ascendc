# 性能调优要点

## 基本原则

1. **减少数据搬运**：尽可能复用 Local Memory 中的数据，减少 GM 访问
2. **搬运计算重叠**：使用双缓冲让 MTE2 和 Vector/Cube 并行
3. **对齐访问**：GM 地址 32 字节对齐，UB 操作 32 字节对齐
4. **减少标量操作**：Scalar 单元相对弱，避免循环内大量 if/else 和变量计算

## 常见优化技巧

| 问题 | 优化方法 |
|------|---------|
| UB 带宽瓶颈 | 减少不必要的 DataCopy，合并小搬运为大搬运 |
| Vector 利用率低 | 单次处理尽量多的元素（充分利用 128 FP16 / 64 FP32 宽度） |
| Cube 利用率低 | 确保 M/N/K 对齐到 16 的倍数，使用 FRACTAL_NZ 格式 |
| 流水线停顿 | 使用 BUFFER_NUM=2 乒乓，减少 SetFlag/WaitFlag 次数 |
| 标量开销大 | 用 Duplicate 替代循环 SetValue，用向量化替代逐元素操作 |

## Profiling 工具

使用 `msprof` 工具进行性能分析：
```bash
msprof --output=./profiling_dir python3 test_npu.py ...
```
关注指标：Vector/Cube 利用率、MTE2 带宽、流水线停顿时间。

## 常见瓶颈定位

| 现象 | 可能原因 |
|------|---------|
| MTE2 利用率 100% 但 Vector 空闲 | 搬运带宽不足，需减少 GM 访问或优化数据复用 |
| Vector 利用率低 | 单次处理元素太少，增大 tile 大小 |
| Cube 利用率低 | M/N/K 未对齐，或数据格式非 FRACTAL_NZ |
| 频繁 SetFlag/WaitFlag 等待 | 流水线设计不合理，调整乒乓策略 |

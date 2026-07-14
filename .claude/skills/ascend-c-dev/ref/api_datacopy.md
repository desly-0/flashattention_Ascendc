# 基础 API 详细参考

## 数据搬运 API

### DataCopy

```cpp
// GM → UB / UB → UB / UB → GM
DataCopy(dstLocal, srcLocal, blockCount);
```
- 同步阻塞式搬运
- blockCount 为**元素个数**（不是字节数）
- 源和目标数据类型必须相同

### Duplicate

```cpp
Duplicate(dstLocal, value, blockCount);
```
- 用常量值填充 LocalTensor
- 替代逐元素 SetValue，性能远优于循环 SetValue
- 适合初始化 mask、清零等操作

### Cast

```cpp
Cast(dstLocal, srcLocal, RoundMode::CAST_ROUND, blockCount);
```
- 数据类型转换
- 常见：`float → half`（softmax 输出转 half）、`half → float`
- `RoundMode::CAST_ROUND` 为四舍五入模式

## 向量计算 API

| API | 等价操作 | 说明 |
|-----|---------|------|
| `Add(dst, src, scalar, len)` | dst = src + scalar | 标量加 |
| `Muls(dst, src, scalar, len)` | dst = src × scalar | 标量乘 |
| `Sub(dst, src, scalar, len)` | dst = src - scalar | 标量减 |
| `Mul(dst, src1, src2, len)` | dst = src1 × src2 | 向量对位乘 |
| `Exp(dst, src, len)` | dst = exp(src) | 指数运算 |
| `Recip(dst, src, len)` | dst = 1/src | 倒数运算 |
| `Max(dst, src, len)` | dst = max(dst, src) | 逐元素取大 |
| `ReduceSum(dst, src, len)` | dst = sum(src) | 归约求和 |

所有向量 API 要求：
- 输入输出 Tensor 必须在 **Unified Buffer (UB)** 中
- 32 字节对齐
- len 为元素个数

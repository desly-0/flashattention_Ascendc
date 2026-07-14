# SoftmaxFlashV2 API 详细参考

## SoftmaxFlashV2

对应 FlashAttention-2 的 online softmax 算法。

### 函数原型

```cpp
template <typename T, bool isUpdate, bool isReuseSource, bool isBasicBlock,
          bool isDataFormatNZ, const SoftmaxConfig& config>
__aicore__ inline void SoftmaxFlashV2(
    const LocalTensor<T>& dstTensor,      // 输出 P (softmax 概率)
    const LocalTensor<T>& expSumTensor,   // 输出 sum (exp 和)
    const LocalTensor<T>& maxTensor,      // 输出 max (行最大值)
    const LocalTensor<T>& srcTensor,      // 输入 S (score = QK^T * scale)
    const LocalTensor<T>& expMaxTensor,   // 输出 exp_max (用于 rescale)
    const LocalTensor<T>& inExpSumTensor, // 输入 insum (上一块的 sum)
    const LocalTensor<T>& inMaxTensor,    // 输入 inmax (上一块的 max)
    const SoftMaxTiling& tiling,          // tiling 参数
    const SoftMaxShapeInfo& softmaxShapeInfo = {});
```

### 模板参数

| 参数 | 说明 |
|------|------|
| `T` | 数据类型（通常 float） |
| `isUpdate` | true = 分块合并模式（bi>0），false = 第一块 |
| `isReuseSource` | true = 复用 src 空间作为 dst |
| `isBasicBlock` | false |
| `isDataFormatNZ` | false (ND 格式) |
| `config` | `SOFTMAX_DEFAULT_CFG` |

### 用法示例（FlashAttention）

```cpp
// 第一块 (bi=0): isUpdate=false
SoftmaxFlashV2<float, false, true, false, false, SOFTMAX_DEFAULT_CFG>(
    sL,       // dst: 输出 P
    smS,      // expSumTensor: 输出 sum (l)
    smM,      // maxTensor: 输出 max (m)
    sL,       // srcTensor: 复用 sL 作为输入 (isReuseSource=true)
    smE,      // expMaxTensor: 输出 exp_max
    smS,      // inExpSumTensor: 输入 sum (未使用)
    smM,      // inMaxTensor: 输入 max (未使用)
    smTl,     // tiling
    smSh);    // softmaxShapeInfo

// 后续块 (bi>0): isUpdate=true
SoftmaxFlashV2<float, true, true, false, false, SOFTMAX_DEFAULT_CFG>(
    sL,       // dst
    ns,       // expSumTensor: 新 sum
    nm,       // maxTensor: 新 max
    sL,       // src
    smE,      // expMax: exp(old_max - new_max)
    ps,       // inExpSum: old sum
    pm,       // inMax: old max
    smTl,     // tiling
    smSh);
```

### 算法原理

```
isUpdate=false:
  x_max   = max(src, axis=-1)
  dst     = exp(src - x_max)
  x_sum   = sum(dst, axis=-1)

isUpdate=true:
  x_max   = max(concat([inmax, src]), axis=-1)
  dst     = exp(src - x_max)
  exp_max = exp(inmax - x_max)
  x_sum   = exp_max * insum + sum(dst, axis=-1)
```

## SoftMaxFlashV2TilingFunc

Kernel 侧动态计算 softmax tiling 参数。

```cpp
SoftMaxShapeInfo smSh{bs*qt, kt, bs*qt, kt};  // {M, N, M_aligned, N_aligned}
auto smTl = SoftMaxFlashV2TilingFunc(
    smSh,           // shape 信息
    4,              // 输入数据类型大小 (float=4)
    4,              // 输出数据类型大小 (float=4)
    Br*D*8,         // local workspace 大小
    false, false, false, false  // isUpdate, isBasicBlock, isNZ, isFlashBrc
);
```

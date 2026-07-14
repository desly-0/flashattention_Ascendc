# Matmul API 详细参考

## Matmul 模板参数

```cpp
// 模板参数：<TypeA, TypeB, TypeC, CubeTilingType>
using MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
//               位置        数据格式       数据类型 是否转置
```

**参数说明**：
- `TPosition::GM` — 数据在 Global Memory
- `CubeFormat::ND` — ND 数据排布（非 NZ）
- `half` — 数据类型（支持 half, float, int8 等）
- `false/true` — B 矩阵是否转置

**典型配置**（FlashAttention）：
```cpp
// bmm1: Q × K^T → S
// Q: [Br, D] half, K^T: [D, Bc] half (transposed), S: [Br, Bc] float
using B1AT = MatmulType<GM, ND, half, false>;
using B1BT = MatmulType<GM, ND, half, true>;   // true = 转置
using B1CT = MatmulType<GM, ND, float>;
using B1MM = Matmul<B1AT, B1BT, B1CT, MatmulType<GM, ND, float>>;

// bmm2: P × V → PV
// P: [Br, Bc] half, V: [Bc, D] half, PV: [Br, D] float
using B2AT = MatmulType<GM, ND, half, false>;
using B2BT = MatmulType<GM, ND, half, false>;  // false = 不转置
using B2CT = MatmulType<GM, ND, float>;
using B2MM = Matmul<B2AT, B2BT, B2CT, MatmulType<GM, ND, float>>;
```

## 核心 API

### REGIST_MATMUL_OBJ

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tiling1, mm2, &tiling2);
```
变长参数，每 3 个一组：(Matmul对象, TCubeTiling指针)，可同时注册多个 Matmul。
注意：不能同时使用 CubeResGroup。

### SetTensorA / SetTensorB
```cpp
mm.SetTensorA(qGlobalTensor);  // [M, K]
mm.SetTensorB(kGlobalTensor);  // [K, N] 或 [N, K]（转置时）
```

### IterateAll
```cpp
// 同步启动: 内部等待计算完成
mm.IterateAll<true>(cTensor);

// 异步启动: needWait=false + isAsync=true
mm.IterateAll<false>(cTensor, 0, false, true, false);
```

### WaitIterateAll
```cpp
mm.WaitIterateAll();  // 等待异步 IterateAll 完成
```

## TCubeTiling 结构体

50 个 int32_t 的结构体 (200 bytes, pack(8))，关键字段：

| 字段 | 说明 |
|------|------|
| `M, N, K` | 矩阵维度 |
| `baseM, baseN, baseK` | 分形基大小（910B: 16×16×16） |
| `singleCoreM, singleCoreN, singleCoreK` | 单核计算维度 |
| `depthA1, depthB1` | L1 预取深度 |
| `stepM, stepN` | 步进大小 |
| `iterateOrder` | 迭代顺序 |
| `usedCoreNum` | 使用的核数 |

## Matmul Tiling 类（Host 侧）

```cpp
MatmulTiling tiling;
tiling.SetShape(input_M, input_N, input_K);  // 设置形状
tiling.SetOrgShape(org_M, org_N, org_K);
tiling.SetAType(...);   // A 矩阵数据类型
tiling.SetBType(...);   // B 矩阵数据类型
tiling.SetCType(...);   // C 矩阵数据类型
auto tilingData = tiling.GetTiling();  // 获取 TCubeTiling 数据
```

## 注意事项

1. **16 字节对齐**：M/N/K 建议对齐到 16 的倍数（910B 分形基 16×16×16）
2. **REGIST_MATMUL_OBJ 与 CubeResGroup 互斥**
3. **CANN 9.0.0 中只能有一个 TPipe 实例**
4. 使用异步模式时需确保 GlobalTensor 数据在整个计算完成前不被修改

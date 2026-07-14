# 流水线编程模型

## 乒乓双缓冲模式

乒乓（Double Buffering）是关键优化手段，让数据搬运和计算重叠执行。

### 典型结构

```cpp
// BUFFER_NUM=2: 用两组 Buffer 交替
TPipe pipe;
TQue<TPosition::VECIN, BUFFER_NUM> queIn;   // 输入队列
TQue<TPosition::VECOUT, BUFFER_NUM> queOut; // 输出队列

// 初始化
pipe.InitBuffer(queIn, BUFFER_NUM, totalSize);
pipe.InitBuffer(queOut, BUFFER_NUM, totalSize);

// 主循环
for (int32_t i = 0; i < totalIter; i++) {
    // 1. 搬入 (GM → UB)
    LocalTensor<T> xLocal = queIn.AllocTensor<T>();
    DataCopy(xLocal, xGm[i * BLOCK_SIZE], BLOCK_SIZE);
    queIn.EnQue(xLocal);  // 入队

    // 2. 计算 (从队列取出)
    LocalTensor<T> xProc = queIn.DeQue<T>();
    // ... 计算逻辑 ...
    queOut.EnQue(xProc);  // 结果入队

    // 3. 搬出 (UB → GM)
    LocalTensor<T> zOut = queOut.DeQue<T>();
    DataCopy(zGm[i * BLOCK_SIZE], zOut, BLOCK_SIZE);
    queOut.FreeTensor(zOut);  // 释放
}
```

## SetFlag / WaitFlag 同步

```cpp
// 同步事件类型
enum class HardEvent {
    V_MTE2,    // Vector 等 MTE2 搬入完成（数据就绪）
    MTE2_V,    // MTE2 通知 Vector 数据已写入 UB
    V_MTE3,    // Vector 通知 MTE3 结果已就绪可搬出
    MTE3_V,    // MTE3 通知 Vector 数据已搬出（UB 空闲）
    MTE3_MTE2, // MTE3 通知 MTE2 可以搬入新数据
    S_MTE2,    // Scalar 发起的 MTE2 同步
    // ...
};

// 使用方式
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);   // 发信号
WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);  // 等信号
```

## 流水线编程建议

1. **尽量让 MTE2 和 Vector/Cube 重叠**：在一轮计算的同时发起下一轮的数据搬入
2. **用 BUFFER_NUM=2 避免流水线停顿**
3. **同步点数不宜过多**：每个同步点都有开销
4. **注意 EVENT_ID 的配对**：SetFlag 和 WaitFlag 必须用同一个 EVENT_ID

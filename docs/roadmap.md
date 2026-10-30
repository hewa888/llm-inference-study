# 大模型推理系统学习路线

## 当前只保留的最小主线

```text
文字 → Token → Token ID → Embedding → Transformer → Logits → Softmax / Sampling → 下一个 Token
```

完成后再逐步扩展：

```text
一个请求 → Prefill / Decode / KV Cache → 多个请求 → Serving Framework 调度
→ Operator / Kernel 执行 → GPU 硬件 → 自动优化系统
```

## 12 周路线

| 周次 | 主题 | 主要产物 |
|---|---|---|
| 1-2 | Transformer 推理基础 | 解释第一个和第二个 Token 如何产生 |
| 3 | Prefill、Decode、KV Cache | KV Cache 计算表 |
| 4 | Latency、Throughput、Batching | Benchmark 指标表 |
| 5 | FlashAttention 与内存层次 | IO-aware 加速解释 |
| 6 | Profiling 与 Roofline | 一页 Profiling Report |
| 7 | Quantization | 数据格式比较表 |
| 8 | Speculative Decoding | 收益与失败条件 |
| 9 | vLLM | Latency-Throughput 数据 |
| 10 | SGLang 与 Prefix Cache | 前缀复用实验 |
| 11 | Kernel、FlashInfer、并行推理 | 系统调用链图 |
| 12 | Hyperloom | 5-8 页系统分析 |

## 分层地图，第二遍材料

> 完成 KV Cache 和 vLLM 后再回读。

| 层次 | 典型对象 | 主要问题 |
|---|---|---|
| 模型架构 | Transformer、Attention、MoE | 模型计算什么 |
| 推理解码算法 | Autoregressive、Speculative Decoding | Token 怎样产生 |
| Serving Framework | vLLM、SGLang、TensorRT-LLM | 多个请求怎样高效执行 |
| 编译与 Runtime | TensorRT、torch.compile、XLA、ROCm | 计算图怎样映射和执行 |
| 算子库 | FlashInfer、AITER、CK、cuBLAS | 常见 Operator 如何高效实现 |
| GPU Kernel | HIP、CUDA、Triton Kernel | 单个计算怎样利用硬件 |
| 硬件 | AMD Instinct、NVIDIA GPU | 计算、内存和通信的物理上限 |
| 自动优化系统 | Hyperloom、GEAK | 怎样自动搜索、修改和验证 |

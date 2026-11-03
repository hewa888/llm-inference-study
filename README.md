# LLM Inference Study

面向大模型推理系统的结构化学习、论文阅读与实验记录仓库。

## 核心问题

> 一个大模型推理请求从进入服务到生成 Token，究竟经历了什么，性能瓶颈会出现在哪里？

## 当前学习顺序

1. Token、Tokenizer、Token ID
2. Embedding Table 与 Embedding Lookup
3. Q、K、V、Self-Attention、Causal Mask 与位置信息
4. Transformer Block：Multi-Head Attention、MLP、Residual、Normalization
5. Hidden State、LM Head 与 Logits
6. Softmax、Sampling 与自回归生成
7. Prefill、Decode 与 KV Cache
8. Latency、Throughput 与 Batching
9. PagedAttention 与 Continuous Batching
10. Quantization
11. Speculative Decoding
12. GPU Kernel、Profiling 与 Roofline
13. vLLM / SGLang 系统结构
14. Hyperloom 如何自动搜索和验证优化

## 目录

- [学习路线](docs/roadmap.md)
- [Week 01 入口](notes/week01/README.md)
- [Week 02 入口](notes/week02/README.md)
- [论文清单](papers/README.md)
- [内容维护约定](CONTRIBUTING.md)

## 仓库结构

```text
docs/                 学习路线与术语地图
notes/weekXX/         每周逐日学习笔记
papers/               论文阅读卡片
experiments/          实验记录
templates/            论文、周报、实验模板
assets/               图片等静态资源
scripts/              本地辅助脚本
```

在 Cursor / VS Code 中按 `Ctrl+Shift+V` 预览 Markdown。

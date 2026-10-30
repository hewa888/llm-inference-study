# LLM Inference Study

面向大模型推理系统的结构化学习、论文阅读与实验记录仓库。

## 核心问题

> 一个大模型推理请求从进入服务到生成 Token，究竟经历了什么，性能瓶颈会出现在哪里？

## 当前学习顺序

1. Token、Tokenizer、Token ID
2. Embedding、Logits、Softmax
3. Q、K、V 与 Self-Attention
4. Transformer Block
5. Prefill、Decode 与 KV Cache
6. Latency、Throughput 与 Batching
7. PagedAttention 与 Continuous Batching
8. Quantization
9. Speculative Decoding
10. GPU Kernel、Profiling 与 Roofline
11. vLLM / SGLang 系统结构
12. Hyperloom 如何自动搜索和验证优化

## 目录

- [学习路线](docs/roadmap.md)
- [Week 01 入口](notes/week01/README.md)
- [论文清单](papers/README.md)
- [内容维护约定](CONTRIBUTING.md)

## 仓库结构

```text
docs/                 学习路线与术语地图
notes/week01/         第一周逐日学习笔记
papers/               论文阅读卡片
experiments/          实验记录
templates/            论文、周报、实验模板
assets/               图片等静态资源
scripts/              本地辅助脚本
```

在 Cursor / VS Code 中按 `Ctrl+Shift+V` 预览 Markdown。

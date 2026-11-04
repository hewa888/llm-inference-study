# Week 01：模型输入和主体

## 本周目标

不查资料也能解释：

> “北京是中国的”这句话，怎样一步步变成每个位置的 Final Hidden State？

```text
Raw Text → Token IDs → Initial Embeddings → Transformer Block × N → Final Hidden States
```

## 每日入口

1. [Day 01：Token、Tokenizer、Token ID](day01-tokenizer.md)
2. [Day 02：Embedding、LM Head 与 Softmax](day02-embedding.md)
3. [Day 03：Q、K、V 与 Self-Attention](day03-qkv-attention.md)
4. [Day 04：Causal Mask 与位置信息](day04-causal-mask-position.md)
5. [Day 05：Multi-Head Attention](day05-multi-head-attention.md)
6. [Day 06：MLP、Residual 与 Normalization](day06-mlp-residual-norm.md)
7. [Day 07：一层和多层 Transformer 总结](day07-transformer-blocks.md)
8. [本周总结](week01-summary.md)

## 本周暂停学习

- Temperature、Sampling 等下一个 Token 的选择方法：放到 [Week 02](../week02/README.md)。LM Head、Logits、Softmax 已在 Day 02 提前学习。
- KV Cache、vLLM、SGLang、FlashAttention、GPU Kernel、训练与反向传播、Hyperloom。

# Week 02：模型输出和生成

## 本周目标

不查资料也能解释：

> “北京是中国的”为什么可以逐步生成“首都”？

## 主线

```text
Final Hidden States      [batch_size, sequence_length, hidden_size]
→ Final Norm
→ LM Head（W: [vocab_size, hidden_size]，logits = h · Wᵀ）
→ Logits                 [batch_size, sequence_length, vocab_size]
→ Temperature → Softmax → Top-k / Top-p → Greedy 或 Sampling
→ 下一个 Token ID
→ Tokenizer Decode
```

> 模型主体生成 Logits，解码策略从 Logits 中选择 Token，这是两件不同的事情。

## 每日计划

每日笔记在开始学习当天再创建。

| 天 | 主题 |
|---|---|
| Day 01 | Initial Embedding 与 Contextual Hidden State |
| Day 02 | Final Hidden State 与 LM Head |
| Day 03 | Logits 与 Vocabulary Projection |
| Day 04 | Softmax 与 Temperature |
| Day 05 | Greedy、Sampling、Top-k、Top-p |
| Day 06 | Autoregressive Generation |
| Day 07 | 完整小模型观察实验 |

Day 01 的核心例子：“苹果”在下面两句话中的初始 Embedding 相同，但经过 Transformer 后的 Hidden State 不同。

```text
我吃了一个苹果
苹果发布了新产品
```

## 预留实验（Day 04 使用）

```python
import torch

candidates = ["首都", "城市", "国家"]
for logits in [
    torch.tensor([3.2, 1.5, -0.4]),
    torch.tensor([3.2, 3.1, 3.0]),
    torch.tensor([10.0, 1.0, -5.0]),
    torch.tensor([1.0, 1.0, 1.0]),
]:
    probabilities = torch.softmax(logits, dim=0)
    print("-" * 40)
    for token, logit, p in zip(candidates, logits, probabilities):
        print(f"{token}: logit={logit.item():.2f}, p={p.item():.4f}")
```

观察：

- 三个 logits 很接近时：
- 一个 logit 远大于其他值时：
- 三个 logits 相同时：

## 是否可以进入 Week 03（KV Cache）

- [ ] 能独立解释 Token 的生成流程。
- [ ] 能解释 Embedding、Attention、MLP、Logits。
- [ ] 能区分模型主体（生成 Logits）与解码策略（选择 Token）。
- [ ] 能解释为什么生成第二个 Token 需要再次执行模型。

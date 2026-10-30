# Day 02：Embedding、Logits、Softmax

## 今日目标

理解两次转换：`Token ID → 向量`，以及 `候选分数 → 概率`。

- **Embedding**：通过 Token ID 查表得到向量。
- **Logits**：模型为词表中每个候选 Token 输出的原始分数。
- **Softmax**：把分数转换为总和为 1 的概率。

## 实验

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

## 我的观察

- 三个 logits 很接近时：
- 一个 logit 远大于其他值时：
- 三个 logits 相同时：

## 今日验收

- [ ] 能解释 Token ID 与 Embedding 的区别。
- [ ] 知道模型输出的是词表的 logits。
- [ ] 知道 Softmax 只负责把分数归一化。
- [ ] 能描述 `Token ID → Embedding → 模型 → Logits → 下一个 Token`。

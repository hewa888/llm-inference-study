# Day 03：Q、K、V 与 Self-Attention

## 今日目标

理解 Attention 如何让当前位置从其他位置收集信息。

- **Query**：当前位置正在寻找什么。
- **Key**：每个位置提供的可匹配表示。
- **Value**：匹配后真正汇总的信息。

Q、K、V 都是输入向量经过不同矩阵投影得到的向量。

## 学习材料

- The Illustrated Transformer：https://jalammar.github.io/illustrated-transformer/
- Dive into Deep Learning：https://d2l.ai/

## 手算实验

```python
import torch

words = ["小王", "杯子", "桌子"]
scores = torch.tensor([0.2, 1.8, 0.7])
weights = torch.softmax(scores, dim=0)

for word, score, weight in zip(words, scores, weights):
    print(word, "score=", score.item(), "weight=", round(weight.item(), 4))
```

## 今日验收

- [ ] 知道 Q、K、V 来自输入向量的线性投影。
- [ ] 知道 Query 与 Key 的比较产生相关性分数。
- [ ] 知道 Softmax 将分数转为 Attention 权重。
- [ ] 知道被加权汇总的是 Value。
- [ ] 能解释 Self-Attention 中的 Self。

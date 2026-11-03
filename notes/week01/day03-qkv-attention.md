# Day 03：Q、K、V 与 Self-Attention

## 今日目标

理解 Attention 如何让当前位置从其他位置收集信息。

- **Query**：当前位置正在寻找什么。
- **Key**：每个位置提供的可匹配表示。
- **Value**：匹配后真正汇总的信息。

Q、K、V 都是输入向量经过不同矩阵投影得到的向量。

## 主线

```text
初始 Embedding               [sequence_length, hidden_size]
→ 三个线性投影得到 Q、K、V    [sequence_length, head_dim]
→ Q · Kᵀ / √head_dim         Attention Score [sequence_length, sequence_length]
→ Softmax（按行）             Attention 权重，每行和为 1
→ 权重 · V                   Attention Output [sequence_length, head_dim]
```

今天开始回答：“北京”的向量怎样结合“是中国的”等上下文信息？

- **Attention Score**：第 i 行第 j 列表示位置 i 的 Query 与位置 j 的 Key 有多匹配。
- **除以 √head_dim**：防止点积随维度变大，导致 Softmax 过于尖锐。
- **Attention Output**：每个位置得到所有位置 Value 的加权平均，这就是“汇总上下文”。

今日暂不学习：位置信息和 Causal Mask（Day 04）、多个头（Day 05）。

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

## 实验：一个完整的单头 Self-Attention

```python
import torch

torch.manual_seed(0)
tokens = ["北京", "是", "中国", "的"]
seq_len, hidden_size, head_dim = 4, 8, 6

x = torch.randn(seq_len, hidden_size)
W_q = torch.randn(hidden_size, head_dim)
W_k = torch.randn(hidden_size, head_dim)
W_v = torch.randn(hidden_size, head_dim)

Q, K, V = x @ W_q, x @ W_k, x @ W_v
scores = Q @ K.T / head_dim ** 0.5
weights = torch.softmax(scores, dim=-1)
output = weights @ V

print("Q / K / V shape:", tuple(Q.shape))
print("scores shape:", tuple(scores.shape))
print("output shape:", tuple(output.shape))
print("每行权重之和:", weights.sum(dim=-1))
for token, row in zip(tokens, weights):
    print(token, [round(w, 3) for w in row.tolist()])
```

## 我的观察

- Q、K、V、scores、output 的 shape 分别是：
- scores 为什么是 `[sequence_length, sequence_length]`：
- “北京”这一行的权重有没有分给它后面的 Token？生成文本时这合理吗（Day 04 回答）：

## 今日验收

- [ ] 知道 Q、K、V 来自输入向量的线性投影。
- [ ] 知道 Query 与 Key 的比较产生相关性分数。
- [ ] 知道 Softmax 将分数转为 Attention 权重。
- [ ] 知道被加权汇总的是 Value。
- [ ] 能解释 Self-Attention 中的 Self。
- [ ] 能写出 Q、K、V、scores、output 的 shape。

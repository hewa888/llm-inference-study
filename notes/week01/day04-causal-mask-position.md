# Day 04：Causal Mask 与位置信息

## 今日目标

给 Day 03 的 Self-Attention 补上两个缺口：不能看未来，以及要知道顺序。

- **Causal Mask**：生成式模型中，位置 i 只能看到位置 0 到 i，不能看到后面的 Token。
- **位置信息**：Attention 本身不区分顺序，必须额外告诉模型每个 Token 在第几个位置。

## Causal Mask

```text
Attention Score [sequence_length, sequence_length]
→ 把未来位置（j > i）的分数设为 -∞
→ Softmax：-∞ 对应的权重变成 0
→ 每个位置只汇总自己和前面位置的 Value
```

生成时模型从左到右逐个产生 Token，生成第 i 个位置时后面的 Token 还不存在，所以计算时也不能让它看到。

## 位置信息

Self-Attention 只比较向量内容，不管它们排在哪里：把输入打乱顺序，每个 Token 的输出不变，只是跟着换了位置。没有位置信息时，“我爱你”和“你爱我”对模型来说是同一组 Token。

常见做法有两种：

- **绝对位置 Embedding**：每个位置有一个向量，直接加到初始 Embedding 上（原始 Transformer、GPT-2）。
- **RoPE（旋转位置编码）**：不改初始 Embedding，而是在每一层 Attention 中按位置旋转 Q 和 K，让 Q·K 的结果与两个 Token 的相对距离有关（Llama、Qwen）。

## 学习材料

- The Illustrated Transformer（Positional Encoding 部分）：https://jalammar.github.io/illustrated-transformer/
- Rotary Embeddings: A Relative Revolution：https://blog.eleuther.ai/rotary-embeddings/

## 实验 1：Causal Mask

```python
import torch

torch.manual_seed(0)
tokens = ["北京", "是", "中国", "的"]
seq_len = len(tokens)
scores = torch.randn(seq_len, seq_len)

mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)

print("mask（True 表示看不到）:\n", mask)
for token, row in zip(tokens, weights):
    print(token, [round(w, 3) for w in row.tolist()])
```

## 实验 2：没有位置信息时，Attention 不知道顺序

```python
import torch

torch.manual_seed(0)
seq_len, hidden_size = 4, 8
x = torch.randn(seq_len, hidden_size)
W_q, W_k, W_v = (torch.randn(hidden_size, hidden_size) for _ in range(3))


def attention(x):
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    weights = torch.softmax(Q @ K.T / hidden_size ** 0.5, dim=-1)
    return weights @ V


perm = torch.tensor([2, 0, 3, 1])
print("打乱顺序后，每个 Token 的输出不变:", torch.allclose(attention(x)[perm], attention(x[perm]), atol=1e-5))
```

## 我的观察

- 加了 Mask 后，“北京”这一行的权重是：
- 权重矩阵是什么形状的三角形：
- 实验 2 输出 True 说明了什么：

## 今日验收

- [ ] 能解释为什么生成式模型需要 Causal Mask。
- [ ] 知道 Mask 是在 Softmax 之前把未来位置的分数设为 -∞。
- [ ] 能解释为什么不加位置信息时，Attention 无法区分 Token 的顺序。
- [ ] 能区分绝对位置 Embedding 与 RoPE 加入位置信息的地方。

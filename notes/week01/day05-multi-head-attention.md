# Day 05：Multi-Head Attention

## 今日目标

理解为什么要把一次 Attention 拆成多个头并行计算。

- **Head**：一组独立的 Q、K、V 投影，各自计算一次 Attention。
- **head_dim**：每个头的向量长度，通常是 `hidden_size / num_heads`。
- **多头的意义**：不同的头可以学到不同的关注模式，例如一个头关注相邻词，另一个头关注主语。
- **输出投影 W_o**：把所有头的结果拼接起来，再投影回 hidden_size。

## 主线与 Shape

```text
x                        [batch_size, sequence_length, hidden_size]
→ 投影并拆成多个头 Q、K、V [batch_size, num_heads, sequence_length, head_dim]
→ Attention Score        [batch_size, num_heads, sequence_length, sequence_length]
→ Causal Mask + Softmax
→ 每个头的输出           [batch_size, num_heads, sequence_length, head_dim]
→ 拼接所有头             [batch_size, sequence_length, hidden_size]
→ 输出投影 W_o           [batch_size, sequence_length, hidden_size]
```

以 Qwen2.5-0.5B 为例：hidden_size = 896，num_heads = 14，所以 head_dim = 896 / 14 = 64。

Qwen2.5 实际使用的是 GQA（Grouped-Query Attention）：14 个 Query 头共享 2 组 K、V。这样可以减少需要缓存的 K、V，Week 03 学 KV Cache 时再细看。

## 学习材料

- The Illustrated Transformer（Multi-Headed Attention 部分）：https://jalammar.github.io/illustrated-transformer/

## 实验：手写 Multi-Head Attention，并与 PyTorch 内置实现对比

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
batch_size, seq_len, hidden_size, num_heads = 1, 4, 24, 3
head_dim = hidden_size // num_heads

x = torch.randn(batch_size, seq_len, hidden_size)
W_q, W_k, W_v, W_o = (torch.randn(hidden_size, hidden_size) / hidden_size ** 0.5 for _ in range(4))


def split_heads(t):
    return t.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


Q, K, V = split_heads(x @ W_q), split_heads(x @ W_k), split_heads(x @ W_v)
scores = Q @ K.transpose(-2, -1) / head_dim ** 0.5
mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
heads = weights @ V
merged = heads.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
output = merged @ W_o

for name, t in [("x", x), ("Q", Q), ("scores", scores), ("heads", heads), ("merged", merged), ("output", output)]:
    print(f"{name:7s}", tuple(t.shape))

reference = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print("与 PyTorch 内置实现一致:", torch.allclose(heads, reference, atol=1e-5))
```

## 我的观察

- 每一步 shape 中的数字分别对应哪个维度：
- `split_heads` 中 `view` 和 `transpose` 各做了什么：
- 为什么 Multi-Head Attention 的输入和输出 shape 相同：

## 今日验收

- [ ] 能解释为什么用多个头，而不是一个大头。
- [ ] 能写出 `head_dim = hidden_size / num_heads`，并算出 Qwen2.5-0.5B 的 head_dim。
- [ ] 能说出 Multi-Head Attention 每一步的 shape 变化。
- [ ] 知道多头输出拼接后还要经过输出投影 W_o。

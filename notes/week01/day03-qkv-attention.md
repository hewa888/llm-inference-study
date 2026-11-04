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

位置信息和 Causal Mask（Day 04）、多个头（Day 05）有单独的笔记，文末“我的总结”只记要点。

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

## 我的结果

```text
Q / K / V shape: (4, 6)
scores shape: (4, 4)
output shape: (4, 6)
每行权重之和: tensor([1., 1., 1., 1.])
北京 [0.928, 0.002, 0.061, 0.01]
是 [0.012, 0.276, 0.21, 0.502]
中国 [0.997, 0.0, 0.0, 0.003]
的 [0.0, 0.0, 1.0, 0.0]
```

## 我的观察

- Q、K、V、scores、output 的 shape 分别是：Q、K、V 都是 `(4, 6)`，scores 是 `(4, 4)`，output 是 `(4, 6)`。4 是 sequence_length，6 是 head_dim。
- scores 为什么是 `[sequence_length, sequence_length]`：每个 Token 的 Query 都要和每个 Token 的 Key 比一次，4 个 Token 就得到 4 × 4 个分数。

  | Query \ Key | 北京 | 是 | 中国 | 的 |
  |---|---|---|---|---|
  | 北京的 Query | score | score | score | score |
  | 是的 Query | score | score | score | score |
  | 中国的 Query | score | score | score | score |
  | 的的 Query | score | score | score | score |

  - 行：当前哪个 Token 正在查询。
  - 列：当前 Query 正在查询哪个 Token。

- “北京”这一行的权重有没有分给它后面的 Token？生成文本时这合理吗（Day 04 回答）：
  - 有。“北京”这一行是 `[0.928, 0.002, 0.061, 0.01]`，约 7% 的权重分给了后面的“是”“中国”“的”。
  - 生成时不合理：推理时后面的 Token 还没生成；训练时看后面等于偷看答案。
  - 所以 Day 04 要加 Causal Mask，把未来位置的权重变成 0。

## 今日验收

- [ ] 知道 Q、K、V 来自输入向量的线性投影。
  - $W_Q$、$W_K$、$W_V$ 是训练得到的权重，推理时固定不变；输入 X 分别乘以这三个矩阵，得到 Q、K、V：$Q = XW_Q$，$K = XW_K$，$V = XW_V$。
  - 线性投影就是“乘一个矩阵”：把每个 Token 的 H 维向量变成 D 维向量。叫“线性”是因为只有乘法和加法，没有激活函数。
  - 这里做的是矩阵乘法，不是单个点积。Q 中每个数都是 X 的一行与 $W_Q$ 的一列做点积，所以矩阵乘法可以看成一批点积。本实验中 X 是 `[4, 8]`，$W_Q$ 是 `[8, 6]`，Q 是 `[4, 6]`，24 个数各是一次长度为 8 的点积。
- [ ] 知道 Query 与 Key 的比较产生相关性分数。
- [ ] 知道 Softmax 将分数转为 Attention 权重。
- [ ] 知道被加权汇总的是 Value。
- [ ] 能解释 Self-Attention 中的 Self。
- [ ] 能写出 Q、K、V、scores、output 的 shape。

## 我的总结

### 心智图

```text
Token IDs → Embedding Lookup → Initial Embeddings（+ 位置信息）
↓
Transformer Block × N
├── Self-Attention
│   ├── Q = XW_Q，K = XW_K，V = XW_V
│   ├── Scores = QKᵀ / √D
│   ├── Causal Mask
│   ├── Weights = Softmax(Scores)
│   └── Output = Weights × V
├── Residual + Norm
├── MLP / FFN
└── Residual + Norm
↓
Final Hidden States → LM Head → Logits → 选出下一个 Token
```

从文本到 Logits 的完整链路见 [Day 02 的“我的总结”](day02-embedding.md)。

### Self-Attention 公式与 Shape

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{D}}\right)V
$$

```text
X                    [S, H]     S = sequence_length，H = hidden_size
W_Q / W_K / W_V      [H, D]     D = head_dim
Q / K / V            [S, D]
Scores / Weights     [S, S]     第 i 行第 j 列：位置 i 的 Query 与位置 j 的 Key 的匹配程度
Output               [S, D]
```

加入 Batch 后，每个 shape 最前面多一维 B。

- **职责**：Q、K 决定看哪里、看多少；V 提供真正被汇总的内容。
- **权重 vs 计算结果**：$W_Q$、$W_K$、$W_V$ 是训练学到的权重，推理时固定不变；Q、K、V 是推理时用这些权重对当前输入现算出来的向量，随输入变化。它们不是人工定义的问题、标签和答案。
- **除以 √D**：维度越大，点积波动越大，Softmax 会过于尖锐、梯度变差；除以 √D 把分数控制在稳定范围。
- **Output**：每一行是所有允许访问位置的 Value 的加权平均。
- **Self**：Q、K、V 都来自同一个序列 X，不是“只关注自己”。

### 完整的 GPT Attention 还需要

- **Causal Mask**：Softmax 前把未来位置的分数设为 -∞，权重变成 0，每个位置只能看自己和左侧。没有 Mask 的实验不是完整的 GPT Attention。
- **位置信息**：Attention 本身不区分顺序。Token 身份来自 Embedding，顺序来自位置信息：GPT-2 用可学习的位置 Embedding，Qwen 用 RoPE 作用于 Q 和 K。
- **Multi-Head**：多组 Q、K、V 投影并行计算，拼接后经输出投影回到 hidden_size。各个头的功能通常是分布式的，不要认为某个头固定负责语法或指代。

### 一层 Transformer Block

以 Pre-Norm 为例（不同模型的顺序可能不同）：

```text
h   = x + Attention(Norm(x))     Attention：跨 Token 交流，沿 Sequence 维度混合信息
out = h + MLP(Norm(h))           MLP：单个 Token 内部计算，沿 Feature 维度加工信息
```

- **MLP**：先升维再降维，例如 Qwen2.5-0.5B 是 896 → 4864 → 896。典型形式为 $\text{MLP}(x) = W_2\,\sigma(W_1 x + b_1) + b_2$。
- **Residual**：$Y = X + \text{Module}(X)$。保留输入，让模块只学习增量修正，帮助深层网络的梯度传播。
- **Normalization**：对每个 Token 的向量单独做，把数值尺度拉回稳定范围。原始 Transformer 用 LayerNorm，Llama、Qwen 用 RMSNorm。
  - **为什么需要**：Residual 不断把各层输出加到主干上，层数一多数值越来越大，会让 Softmax 过于尖锐、训练时梯度不稳定。
  - **做了什么**：RMSNorm 只缩放向量的整体大小，不改变各维度的比例，例如 `[1, 2, 3]` 和 `[10, 20, 30]` 都变成约 `[0.46, 0.93, 1.39]`，再乘可学习的 $\gamma$。
  - **放在哪里**：Attention 和 MLP 之前，即子层的输入处。所有层结束后、LM Head 之前还有最后一次。
  - **只看单个 Token**：只沿 hidden_size 计算，不跨 Token，也不跨 batch。

$$
\text{LayerNorm}(x)_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i,
\qquad \mu = \frac{1}{d}\sum_{j=1}^{d} x_j,
\qquad \sigma^2 = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu)^2
$$

$$
\text{RMSNorm}(x)_i = \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}}
$$

$d$ 是 hidden_size，$\gamma$、$\beta$ 是可学习参数，$\epsilon$ 防止除以 0（Qwen 中为 $10^{-6}$）。RMSNorm 去掉了减均值和 $\beta$，计算更省。

### 容易混淆的概念

- **Initial Embedding vs Hidden State**：前者直接查表得到，与上下文无关；后者经过 Transformer，结合了上下文。“我吃了一个苹果”和“苹果发布了新产品”中，“苹果”的初始 Embedding 相同，Hidden State 不同。
- **Attention Score vs Weight**：Score 是 Q 与 K 的原始匹配分数；Weight 是经过 Mask 和 Softmax 后的结果。
- **Key vs Value**：Key 用来决定是否匹配；Value 是匹配后被加权汇总的内容。
- **Attention Softmax vs 词表 Softmax**：数学操作相同，前者把 Attention Scores 变成位置权重，后者把 Logits 变成 Token 概率。
- **Residual vs KV Cache**：Residual 是层内把输入加回输出；KV Cache 是推理时保存历史 Token 的 K、V，避免重复计算。

### 提醒

- Attention Weight 高，不等于人类能直接解释的语义关系。
- 单头实验用于理解公式，真实模型使用 Multi-Head Attention。
- 学习时区分通用结构和具体实现，不要假设所有模型的 Block 顺序都相同。

### 一句话总结

Token ID 查表得到初始向量；Self-Attention 用 Q、K 计算各位置之间的权重，再按权重混合 V，让每个 Token 获得上下文信息；MLP 对每个位置进一步加工；多层 Transformer Block 逐层形成上下文化的 Hidden State；最后 LM Head 把最后一个位置的 Hidden State 映射成整个词表的 Logits，用于选择下一个 Token。

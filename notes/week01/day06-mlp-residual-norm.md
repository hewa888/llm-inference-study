# Day 06：MLP、Residual 与 Normalization

## 今日目标

认识一层 Transformer 中除 Attention 以外的三个组件。

- **MLP / FFN**：对每个 Token 的向量独立加工，先升维到 intermediate_size，再降回 hidden_size。Token 之间不交换信息。
- **Residual Connection**：子层的输出加回输入，即 `x + Sublayer(x)`。子层只需要学习“增量”，原信息不会丢失，深层网络也更容易训练。
- **Normalization**：把每个 Token 的向量拉回稳定的数值尺度。
  - **LayerNorm**：减去均值、除以标准差，再乘以可学习的缩放、加上偏移（原始 Transformer、GPT-2）。
  - **RMSNorm**：不减均值，只除以均方根，再乘以可学习的缩放，计算更省（Llama、Qwen）。
- **Pre-Norm**：先 Norm 再进入子层，即 `x + Sublayer(Norm(x))`。现在的大模型基本都用这种顺序。

## Attention 与 MLP 的分工

```text
Attention：Token 之间交换信息（看其他位置）
MLP：     每个 Token 独立加工收集到的信息（只看自己）
```

以 Qwen2.5-0.5B 为例：hidden_size = 896，intermediate_size = 4864。它的 MLP 是 SwiGLU 结构：

```text
MLP(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

## 学习材料

- GLU Variants Improve Transformer（SwiGLU）：https://arxiv.org/abs/2002.05202
- Root Mean Square Layer Normalization（RMSNorm）：https://arxiv.org/abs/1910.07467

## 实验

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
seq_len, hidden_size, intermediate_size = 4, 8, 32
x = torch.randn(seq_len, hidden_size) * 3 + 1

layer_norm = (x - x.mean(-1, keepdim=True)) / x.std(-1, unbiased=False, keepdim=True)
rms_norm = x / x.pow(2).mean(-1, keepdim=True).sqrt()
print("原始每行均方根:", x.pow(2).mean(-1).sqrt())
print("LayerNorm 后每行均值:", layer_norm.mean(-1))
print("RMSNorm 后每行均方根:", rms_norm.pow(2).mean(-1).sqrt())

W_gate = torch.randn(hidden_size, intermediate_size) / hidden_size ** 0.5
W_up = torch.randn(hidden_size, intermediate_size) / hidden_size ** 0.5
W_down = torch.randn(intermediate_size, hidden_size) / intermediate_size ** 0.5


def mlp(h):
    return (F.silu(h @ W_gate) * (h @ W_up)) @ W_down


print("MLP 输出 shape:", tuple(mlp(rms_norm).shape))

changed = rms_norm.clone()
changed[1:] = torch.randn(seq_len - 1, hidden_size)
print("改掉其他 Token 后，第 0 个 Token 的 MLP 输出不变:", torch.allclose(mlp(rms_norm)[0], mlp(changed)[0]))

y = x + mlp(rms_norm)
print("Residual 后 shape:", tuple(y.shape))
```

## 我的观察

- LayerNorm 和 RMSNorm 处理后，每行的数值各有什么特点：
- “第 0 个 Token 的 MLP 输出不变”说明了什么：
- 为什么 Residual 要求子层输入和输出的 shape 相同：

## 今日验收

- [ ] 能区分 Attention（Token 之间交换信息）与 MLP（每个 Token 独立加工）。
- [ ] 能写出 Residual 的形式 `x + Sublayer(x)`，并说出它的作用。
- [ ] 能区分 LayerNorm 与 RMSNorm。
- [ ] 知道 Pre-Norm 是把 Norm 放在子层之前。

# Day 07：一层和多层 Transformer 总结

## 今日目标

把 Day 03 到 Day 06 的组件组装成一层 Transformer，再把多层堆叠起来。

## 一层 Transformer Block

以 Llama、Qwen 使用的 Pre-Norm 结构为例：

```text
输入隐藏向量
→ Norm
→ Multi-Head Self-Attention（含 Causal Mask 与位置信息）
→ Residual Addition
→ Norm
→ MLP / FFN
→ Residual Addition
→ 输出到下一层
```

- **Attention**：Token 之间交换和汇总信息。
- **MLP / FFN**：每个 Token 独立加工收集到的信息。
- **Residual**：保留原信息，并改善深层网络中的信息与梯度传递。
- **Norm（LayerNorm / RMSNorm）**：稳定隐藏向量的数值尺度。
- **Multi-Head**：允许多组不同投影和关注模式并行工作。

## 多层堆叠

```text
Token IDs                 [batch_size, sequence_length]
→ Embedding Lookup
Initial Embeddings        [batch_size, sequence_length, hidden_size]
→ Transformer Block 1
→ Transformer Block 2
→ ...
→ Transformer Block N
Final Hidden States       [batch_size, sequence_length, hidden_size]
```

每一层的输入和输出 shape 相同，所以可以一层接一层地堆叠。Final Hidden States 是 [Week 02](../week02/README.md) 的起点：LM Head 会把它变成 Logits。

## 实验：观察 Qwen2.5-0.5B 的真实结构

`torch.device("meta")` 只创建模型结构、不分配内存，所以这里只下载很小的 `config.json`，不下载约 1 GB 的权重。

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
config = AutoConfig.from_pretrained(model_name)

for key in [
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "tie_word_embeddings",
]:
    print(f"{key}: {getattr(config, key)}")

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

print(model)
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M")
```

## 我的观察

把打印出的模块对应到本周学过的概念：

| 模块 | 对应概念 | 第几天学的 |
|---|---|---|
| `embed_tokens` |  |  |
| `q_proj` / `k_proj` / `v_proj` |  |  |
| `o_proj` |  |  |
| `rotary_emb` |  |  |
| `mlp`（`gate_proj` / `up_proj` / `down_proj`） |  |  |
| `input_layernorm` / `post_attention_layernorm` |  |  |
| `layers (0-23)` |  |  |
| `norm` / `lm_head` |  |  |

- 参数量和模型名字中的 0.5B 对得上吗：
- `q_proj` 输出 896，`k_proj` 和 `v_proj` 却只输出 128，为什么（提示：`num_key_value_heads`）：

## 今日验收

- [ ] 能按顺序描述一层 Transformer 的组件。
- [ ] 能区分 Attention 与 MLP 的职责。
- [ ] 知道层与层之间传递的是隐藏向量，shape 保持 `[batch_size, sequence_length, hidden_size]`。
- [ ] 能把 Qwen2.5-0.5B 的每个模块对应到本周学过的概念。

## 仍然不理解的地方

1. 
2. 
3. 

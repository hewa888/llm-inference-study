# Day 04：Transformer Block

## 今日目标

把前三天的概念组装成一层 Transformer。

```text
输入 Token 向量
→ LayerNorm
→ Self-Attention
→ Residual Addition
→ LayerNorm
→ MLP / FFN
→ Residual Addition
→ 输出到下一层
```

- **Attention**：Token 之间交换和汇总信息。
- **MLP / FFN**：每个 Token 独立加工收集到的信息。
- **Residual**：保留原信息，并改善深层网络中的信息与梯度传递。
- **LayerNorm**：稳定隐藏向量的数值尺度。
- **Multi-Head**：允许多组不同投影和关注模式并行工作。

## 今日验收

- [ ] 能按顺序描述一层 Transformer 的组件。
- [ ] 能区分 Attention 与 MLP 的职责。
- [ ] 知道层与层之间传递的是隐藏向量。

## 仍然不理解的地方

1. 
2. 
3. 

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
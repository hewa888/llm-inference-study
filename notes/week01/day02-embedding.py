import torch
from transformers import AutoConfig, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("vocab_size:", config.vocab_size)
print("len(tokenizer):", len(tokenizer))
print("hidden_size:", config.hidden_size)

input_ids = tokenizer("北京是中国的首都", return_tensors="pt").input_ids
print("input_ids:", input_ids)
print("input_ids.shape:", tuple(input_ids.shape))

embedding_table = torch.nn.Embedding(config.vocab_size, config.hidden_size)
print("embedding_table.weight.shape:", tuple(embedding_table.weight.shape))

embeddings = embedding_table(input_ids)
print("embeddings.shape:", tuple(embeddings.shape))

first_id = input_ids[0, 0]
print("查表 = 取第 token_id 行:", torch.equal(embeddings[0, 0], embedding_table.weight[first_id]))

one_hot = torch.nn.functional.one_hot(input_ids, num_classes=config.vocab_size).float()
print("one_hot.shape:", tuple(one_hot.shape))
print("One-hot × Embedding Table = 查表:", torch.allclose(one_hot @ embedding_table.weight, embeddings))
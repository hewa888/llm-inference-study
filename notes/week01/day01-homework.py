from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = [
    "北京是中国的首都",
    "大模型推理",
    "GPU kernel optimization",
    "你好，world!",
]

for text in texts:
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    readable_tokens = [
    tokenizer.convert_tokens_to_string([token])
    for token in tokens
]
    print("=" * 50)
    print("原文:", text)
    print("Tokens:", tokens) #这可能是个编码，人不可读那种
    print("可读 Tokens:", readable_tokens)
    print("Token IDs:", ids)
    print("Token 数量:", len(ids))
    print("还原:", tokenizer.decode(ids))
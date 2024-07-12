import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 配置量化
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 加载模型和分词器
model_name = "shenzhi-wang/Llama3-8B-Chinese-Chat"
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 确保输入张量移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试推理
inputs = tokenizer("你好，世界！", return_tensors="pt").to(device)
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

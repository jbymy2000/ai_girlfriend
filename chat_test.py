from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat")
model = AutoModelForCausalLM.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat")

def chat_with_model(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 测试交互
input_text = "你好，今天天气怎么样？"
response = chat_with_model(input_text)
print(f"模型回答: {response}")

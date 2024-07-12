import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch

# 配置量化
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 加载模型和分词器
model_name = "shenzhi-wang/Llama3-8B-Chinese-Chat"
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 确保输入张量移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_text(prompt):
    # 定义完整的对话 prompt
    full_prompt = f"你好，我是我的女朋友，以下是我们的对话内容：\n{prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=150,  # 根据需要可以调整生成文本的长度
        no_repeat_ngram_size=2,
        temperature=0.9,
        top_k=40,
        top_p=0.9,
        num_return_sequences=1
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.strip()

# 创建 Gradio 界面
iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Type something..."),
    outputs=gr.Textbox(),
    title="Text Generation with llama-3",
    description="Type in a prompt and see how llama3 completes it!"
)

# 运行界面
iface.launch()

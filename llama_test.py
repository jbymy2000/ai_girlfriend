# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat")
model = AutoModelForCausalLM.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat")
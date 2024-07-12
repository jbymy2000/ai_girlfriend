import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

# 配置量化
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 加载模型和分词器
model_name = "shenzhi-wang/Llama3-8B-Chinese-Chat"
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
print(f"EOS Token ID: {eos_token_id}, Pad Token ID: {pad_token_id}")
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0,128009]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
                for item in history_transformer_format])

    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=0.6,
        num_beams=1,
        eos_token_id=10174,   
        pad_token_id=10174,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        print(f"New token: {new_token}")
        if new_token != '<':
            partial_message += new_token
            # 打印当前生成的所有token ID
            yield partial_message

gr.ChatInterface(predict).launch()
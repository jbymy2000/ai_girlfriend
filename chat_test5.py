import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

# 配置量化
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 加载模型和分词器
model_name = "shenzhi-wang/Llama3-8B-Chinese-Chat"
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
newline_token_id = tokenizer(":", return_tensors="pt")["input_ids"][0][0].item()
print(f"EOS Token ID: {eos_token_id}, Pad Token ID: {pad_token_id}, Newline Token ID: {newline_token_id}")


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token_id = input_ids[0][-1].item()
        last_token = tokenizer.decode([last_token_id])
        print(f"Token ID: {last_token_id}, Token: {last_token}")
        stop_ids = [524]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(["".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])
                        for item in history_transformer_format])

    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=False)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=0.6,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )

    generated_tokens = []

    def generate_with_streaming():
        output = model.generate(**generate_kwargs)
        for token_id in output[0].tolist():
            token = tokenizer.convert_ids_to_tokens(token_id)
            generated_tokens.append((token, token_id))

    t = Thread(target=generate_with_streaming)
    t.start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message


gr.ChatInterface(predict).launch()

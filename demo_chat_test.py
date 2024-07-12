import gradio as gr
from chat_test1 import init_model,device,terminators,user_avatar,bot_avatar
from threading import Thread
import random

with gr.Blocks() as demo:
    # step1: 载入模型
    init_model()

    # step2: 初始化gradio的chatbot应用，并添加按钮等信息 /home/rex/Documents/Codes/ai_girlfriend/avatars
    chatbot = gr.Chatbot(
        height=900,avatar_images=(user_avatar, bot_avatar)
    )
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    # 清楚历史记录
 
    def clear_history():
        global llama3_chat_history
        llama3_chat_history = []


    # 用于回复的方法
    def respond(message, chat_history):

        # 引入全局变量
        global llama3_chat_history, tokenizer, model, streamer

        # 拼接对话历史
        llama3_chat_history.append({"role": "user", "content": message})

        # 使用Llama3自带的聊天模板，格式化对话记录
        history_str = tokenizer.apply_chat_template(
            llama3_chat_history,
            tokenize=False,
            add_generation_prompt=True
        )

        # 对历史记录进行tokenization
        inputs = tokenizer(history_str, return_tensors='pt').to(device)

        # 这个历史记录是Gradio的Chatbot自带的变量，用来控制页面显示逻辑的，我们必须也要对齐操作，保证页面展示正常
        chat_history.append([message, ""])

        # 拼接推理参数
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
            num_beams=1,
            do_sample=True,
            top_p=0.8,
            temperature=0.3,
            eos_token_id=terminators
        )

        # 启动线程，用以监控流失输出结果
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 循环推理streamer，每次得到新增的推理部分都附到历史记录末尾，Gradio会监控这个变量在页面展示
        for new_text in streamer:
            chat_history[-1][1] += new_text
            yield "", chat_history

 # 所有的输出完毕之后，我们自己的历史记录也要更新，把模型输出的完整结果加进来。
        llama3_chat_history.append(
            {"role": "assistant", "content": chat_history[-1][1]}
        )
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7800)
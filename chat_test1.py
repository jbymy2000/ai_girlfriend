from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig,TextIteratorStreamer
import torch
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model_path="shenzhi-wang/Llama3-8B-Chinese-Chat"

bot_avatar = "/home/rex/Documents/Codes/ai_girlfriend/avatars/rick.jpg" # 聊天机器人头像位置
user_avatar = "/home/rex/Documents/Codes/ai_girlfriend/avatars/morty.jpg" 

# 存储全局的历史对话记录，Llama3支持系统prompt，所以这里默认设置！
llama3_chat_history = [
 {"role": "system", "content": "You are a helpful assistant trained by MetaAI! But you are running with DataLearnerAI Code."}
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化所有变量，用于载入模型
tokenizer = None
streamer = None
model = None
terminators = None

def init_model():
    """初始化模型，载入本地模型
    """
    global tokenizer, model, streamer, terminators
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True, quantization_config=quantization_config
 )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
 ]

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
 )



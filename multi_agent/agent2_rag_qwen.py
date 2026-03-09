import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from agent1_intent_classifier import get_intent, load_and_sample_examples

# ================= 显存优化 =================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# ================= 路径配置 =================
PROJECT_ROOT = "/tmp/pycharm_project_769/MentalHealth-Counselling-System"
BASE_QWEN_PATH = "/autodl-fs/data/models/Qwen2.5-7B-Instruct"  # 完整基座模型
FINETUNED_QWEN_PATH = "/autodl-fs/data/models/second-qwen-finetuned"  # 队友 LoRA 适配器
EMBED_PATH = "/autodl-fs/data/models/llama-embed-nemotron-8b"
FAISS_INDEX_PATH = "/tmp/pycharm_project_769/MentalHealth-Counselling-System/baseline_v2_llama/faiss_index_llama"
DATA_FILE = os.path.join(PROJECT_ROOT, "1040data_full_labeled_ALL.jsonl")

# ================= tokenizer（优先用适配器的 tokenizer，包含 chat_template） =================
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_QWEN_PATH, trust_remote_code=True)

# 预加载 Few-shot 示例
try:
    print(f"正在从 {DATA_FILE} 加载意图分类示例...")
    GLOBAL_EXAMPLES = load_and_sample_examples(DATA_FILE, num_per_class=4)
except Exception as e:
    print(f"警告：加载示例失败，将使用兜底逻辑。错误: {e}")
    GLOBAL_EXAMPLES = []

# ================= 生成模型（Qwen2.5 + LoRA） =================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_QWEN_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

base_model.config.use_cache = False

model = PeftModel.from_pretrained(
    base_model,
    FINETUNED_QWEN_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    is_trainable=False
)

# Qwen 生成 pipeline（加 stop_strings 防止续写 user）


llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=384,
    temperature=0.7,
    do_sample=False,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# ================= 嵌入模型 =================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_PATH,
    model_kwargs={
        "local_files_only": True,
        "device": "cuda",
        "trust_remote_code": True
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 4
    }
)

# ================= FAISS =================
db = FAISS.load_local(
    folder_path=FAISS_INDEX_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ================= 内存（使用 ChatMessageHistory） =================
chat_history = ChatMessageHistory()

# ================= Prompt =================
prompts = {
    "1": PromptTemplate(
        input_variables=["history", "query", "context"],
        template="""[INST] 你是一个温暖、共情的心理倾听者。现在用户在倾诉情绪，只需要被理解和陪伴。

输出要求：
- 只输出纯回复文本，禁止输出任何其他内容。
- 禁止出现“用户当前输入”“来访者”“如果”“原则”“我们可以尝试”“你愿意试试”“认知行为疗法”“脱敏疗法”“森田疗法”等任何词。
- 禁止复述、改写、重复用户原话。
- 禁止编造用户没说过的话、动作、场景、治疗建议。
- 禁止生成任何形式的后续对话、下一句、示例。
- {context} 禁止在输出中出现、提及、列出、引用。
- 禁止重复任何句子或段落。
- 回复必须完全基于 {query} 和 {history}。
- 用自然、温暖语气表达共情和陪伴。
- 字数 500 字，只回复一次。

真实历史：{history}
用户输入：{query}

回复： [/INST]"""
    ),

    "2": PromptTemplate(
        input_variables=["history", "query", "context"],
        template="""[INST] 你是一个专业的心理咨询师。现在用户在寻求建议阶段，需要实用指导。

输出要求：
- 只输出纯建议文本，禁止输出任何其他内容。
- 禁止出现“用户当前输入”“来访者”“如果”“原则”“我们可以尝试”“你愿意试试”“认知行为疗法”“脱敏疗法”“森田疗法”等任何词。
- 禁止复述、改写、重复用户原话。
- 禁止编造用户没说过的话、动作、场景、治疗建议。
- 禁止生成任何形式的后续对话、下一句、示例。
- {context} 禁止在输出中出现、提及、列出、引用。
- 禁止重复任何句子或段落。
- 回复必须基于 {query} 和 {history}。
- 用自然、专业语气，先简短共情，再给出清晰步骤化建议（用编号）。
- 字数 500 字，只回复一次。

真实历史：{history}
用户问题：{query}

建议： [/INST]"""
    ),

    "3": PromptTemplate(
        input_variables=["history", "query", "context"],
        template="""[INST] 你是一个高度警觉的危机干预咨询师。现在用户可能处于高风险状态，安全第一。

输出要求：
- 只输出纯回复文本，禁止输出任何其他内容。
- 禁止出现“用户当前输入”“来访者”“如果”“原则”“我们可以尝试”“你愿意试试”“认知行为疗法”“脱敏疗法”“森田疗法”等任何词。
- 禁止复述、改写、重复用户原话。
- 禁止编造用户没说过的话、动作、场景。
- 禁止生成任何形式的后续对话、下一句、示例。
- {context} 禁止在输出中出现、提及、列出、引用。
- 禁止重复任何句子或段落。
- 立即表达关心，强烈建议拨打专业热线（北京心理危机干预热线 800-810-1117 / 010-82951332，或当地急救120）。
- 字数 500 字，只回复一次。

真实历史：{history}
用户表达：{query}

回复： [/INST]"""
    )
}

# ================= Agent2 生成函数（使用 messages + apply_chat_template） =================
# def agent2_generate(intent: str, query: str) -> str:
#     if intent not in prompts:
#         return "抱歉，我暂时无法理解您的意图。"
#
#     # 取最近4轮历史
#     history_msgs = chat_history.messages[-4:]
#     history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])
#
#     # RAG 检索
#     docs = retriever.invoke(query)
#     context = "\n\n".join([doc.page_content[:500] for doc in docs])
#
#     # 构建 system prompt
#     system_prompt = prompts[intent].format(history=history_str, query=query, context=context)
#
#     # 使用 Qwen 的 messages 格式
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": query}
#     ]
#
#     # # apply_chat_template 生成输入
#     # input_ids = tokenizer.apply_chat_template(
#     #     messages,
#     #     tokenize=True,
#     #     add_generation_prompt=True,
#     #     return_tensors="pt"
#     # ).to("cuda")
#     #
#     # # 生成
#     # output_ids = model.generate(
#     #     input_ids,
#     #     max_new_tokens=384,
#     #     temperature=0.7,
#     #     do_sample=False,
#     #     stop_strings=["<|im_end|>", "<|im_start|>user"]
#     # )
#
#     # apply_chat_template 生成输入
#     inputs = tokenizer.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_tensors="pt",
#         return_dict=True  # 明确返回字典
#     ).to("cuda")
#
#     # 生成时，解包字典或者直接传 input_ids
#     output_ids = model.generate(
#         input_ids=inputs["input_ids"],  # 修改这里：显式指定 input_ids
#         attention_mask=inputs["attention_mask"],  # 建议带上 mask，更稳定
#         max_new_tokens=384,
#         do_sample=True,  # 既然传了 temperature，这里必须为 True
#         temperature=0.7,
#         top_p=0.9,  # 增加多样性
#         stop_strings=["<|im_end|>", "<|im_start|>user"],
#         tokenizer=tokenizer,
#         pad_token_id=tokenizer.eos_token_id  # 防止 padding 报错
#     )
#
#     response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     # 提取 assistant 部分
#     response = response.split("<|im_start|>assistant\n")[-1].strip()
#
#     # 保存到 chat_history
#     chat_history.add_user_message(query)
#     chat_history.add_ai_message(response)
#
#     # 清洗
#     response = response.strip()
#     black_list = [
#         "用户当前输入", "来访者当前输入", "回复原则", "如果用户", "我们可以尝试"
#     ]
#     for phrase in black_list:
#         if phrase in response:
#             response = response.split(phrase)[0].strip()
#
#     if response.startswith(query):
#         response = response[len(query):].lstrip('，。！？ \n').strip()
#
#         # return response.strip()
#     response = response.strip()
#
#     # 黑名单截断（任何出现就砍掉后面）
#     black_list = [
#         "用户当前输入", "来访者当前输入", "回复原则", "如果用户", "我们可以通过"
#     ]
#     for phrase in black_list:
#         if phrase in response:
#             response = response.split(phrase)[0].strip()
#
#     # 去掉复述开头
#     if response.startswith(query):
#         response = response[len(query):].lstrip('，。！？ \n').strip()
#
#     # 强制截断在句号
#     if len(response) > 300:
#         response = response[:300].rsplit('。', 1)[0] + '。'
#
#     return response

def agent2_generate(intent: str, query: str) -> str:
    """
    生成 AI 回复的核心函数。
    修复了 model.generate 传入字典导致的 AttributeError 报错。
    """
    if intent not in prompts:
        return "抱歉，我暂时无法理解您的意图。"

    # 1. 准备上下文与检索内容
    # 取最近 4 轮历史记录
    history_msgs = chat_history.messages[-4:]
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])

    # RAG 检索知识库内容
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content[:500] for doc in docs])

    # 2. 构建符合意图的 System Prompt
    # 填充 Prompt 模板
    system_prompt_text = prompts[intent].format(
        history=history_str,
        query=query,
        context=context
    )

    # 3. 构造标准对话消息结构
    messages = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": query}
    ]

    # 4. 获取模型输入并强制转换为字典格式（用于后续显式提取）
    # .to("cuda") 会将字典内所有 tensor 移至显存
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True  # 确保返回格式包含 input_ids 和 attention_mask
    ).to("cuda")

    # 5. 【核心修复步】显式提取张量，避免 .shape 缺失报错
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    # 6. 执行文本生成
    # 使用关键字传参 (input_ids=...) 是最稳定的方式
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=384,
        do_sample=True,          # 心理咨询建议开启采样，增加语气自然度
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,  # 惩罚项，防止 AI 陷入循环复读
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # 7. 解码与 Prompt 截断
    # 只解码生成的新内容，跳过输入的 Prompt 长度，防止输出“输出要求”等指令文本
    input_len = input_ids.shape[1]
    response_ids = output_ids[0][input_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # 8. 记录对话历史并清理杂质
    chat_history.add_user_message(query)
    chat_history.add_ai_message(response)

    # 最后的兜底清洗：如果 AI 还是复读了用户的 Query，则切除
    if response.startswith(query):
        response = response[len(query):].lstrip('，。！？ \n').strip()

    return response

# ================= 总 pipeline =================
def consultation_pipeline(user_query: str) -> str:
    history_msgs = chat_history.messages[-4:]
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])

    intent = get_intent(history=history_str, query=user_query, examples=GLOBAL_EXAMPLES)
    print(f"[路由] 检测到意图: {intent}")

    reply = agent2_generate(intent, user_query)
    return reply

# ================= 测试 =================
if __name__ == "__main__":
    print("=== 多轮心理咨询测试（Qwen2.5 版）（输入 quit 退出） ===")

    while True:
        user_input = input("你: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        reply = consultation_pipeline(user_input)
        print(f"AI: {reply}\n")
        print("-" * 80)
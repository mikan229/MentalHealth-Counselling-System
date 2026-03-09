# import os
# import torch
# import faiss
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
# from peft import PeftModel
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_community.llms import HuggingFacePipeline
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.memory import ConversationBufferMemory
#
# from agent1_intent_classifier import get_intent, load_and_sample_examples
#
# # ================= 显存优化（减少碎片） =================
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# torch.cuda.empty_cache()
#
# # ================= 路径配置 =================
# PROJECT_ROOT = "/tmp/pycharm_project_769/MentalHealth-Counselling-System"
# BASE_MISTRAL_PATH = "/autodl-fs/data/models/Mistral-7B-Instruct-v0.3"
# FINETUNED_ADAPTER_PATH = "/autodl-fs/data/models/tuned_model"
# EMBED_PATH = "/autodl-fs/data/models/llama-embed-nemotron-8b"
# FAISS_INDEX_PATH = "/tmp/pycharm_project_769/MentalHealth-Counselling-System/baseline_v2_llama/faiss_index_llama"
# DATA_FILE = os.path.join(PROJECT_ROOT, "1040data_full_labeled_ALL.jsonl")
# torch_dtype = torch.float16
#
# # ================= tokenizer =================
# tokenizer = AutoTokenizer.from_pretrained(BASE_MISTRAL_PATH, trust_remote_code=True)
#
# # 2. 预加载 Few-shot 示例 (修复 ValueError)
# try:
#     print(f"正在从 {DATA_FILE} 加载意图分类示例...")
#     GLOBAL_EXAMPLES = load_and_sample_examples(DATA_FILE, num_per_class=4)
# except Exception as e:
#     print(f"警告：加载示例失败，将使用兜底逻辑。错误: {e}")
#     GLOBAL_EXAMPLES = []
#
# # ================= 生成模型（只在这里 device_map=auto） =================
# # quant_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_compute_dtype=torch.float16,
# #     bnb_4bit_use_double_quant=True,
# # )
#
# # base_model = AutoModelForCausalLM.from_pretrained(
# #     BASE_MISTRAL_PATH,
# #     quantization_config=quant_config,
# #     device_map="auto",
# #     torch_dtype=torch_dtype,
# #     low_cpu_mem_usage=True,
# #     trust_remote_code=True,
# # )
#
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MISTRAL_PATH,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# )
#
# # 关闭 KV cache 减显存
# base_model.config.use_cache = False
#
# model = PeftModel.from_pretrained(
#     base_model,
#     FINETUNED_ADAPTER_PATH,
#     torch_dtype=torch_dtype,
#     is_trainable=False
# )
#
# # ❌ 不要在 pipeline 里再写 device_map
# llm_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=384,   # ↓ 从512降到384
#     temperature=0.7,
#     do_sample=False,
#     return_full_text=False,
# )
# # llm_pipeline = pipeline(
# #     "text-generation",
# #     model=model,
# #     tokenizer=tokenizer,
# #     max_new_tokens=300,  # 限制长度，避免无限续写
# #     temperature=0.05,    # 极低，减少随机脑补
# #     top_p=0.9,
# #     do_sample=False,     # 关闭采样，用贪婪解码（最稳定、最少幻觉）
# #     repetition_penalty=1.3,  # 惩罚重复词句
# #     device_map="auto"
# # )
#
# llm = HuggingFacePipeline(pipeline=llm_pipeline)
#
# # ================= 嵌入模型（强制单卡 fp16） =================
# embeddings = HuggingFaceEmbeddings(
#     model_name=EMBED_PATH,
#     model_kwargs={
#         "local_files_only": True,
#         "device": "cuda",
#         "trust_remote_code": True
#     },
#     encode_kwargs={
#         "normalize_embeddings": True,
#         "batch_size": 4
#     }
# )
#
#
# # ================= FAISS =================
# db = FAISS.load_local(
#     folder_path=FAISS_INDEX_PATH,
#     embeddings=embeddings,
#     allow_dangerous_deserialization=True
# )
#
# # index = faiss.read_index(f"{FAISS_INDEX_PATH}/index.faiss")
# #
# # db = FAISS(
# #     embedding_function=embeddings,
# #     index=index,
# #     docstore=InMemoryDocstore({}),
# #     index_to_docstore_id={}
# # )
#
# retriever = db.as_retriever(search_kwargs={"k": 3})   # ↓ 从4降到3
#
# # ================= 内存 =================
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#
# # ================= Prompt =================
# prompts = {
#     "1": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""[INST] 你是一个温暖、共情的心理倾听者。现在用户在倾诉情绪，只需要被理解和陪伴，不要急于解决问题或给出建议。
#
# 【绝对严格规则，必须100%遵守】
# - 只输出一次纯回复文本，严禁输出任何系统提示、角色标签、示例、解释、推理过程、思考过程。
# - 严禁复述用户原话、改写用户输入、添加“用户说”“来访者说”等。
# - 严禁编造用户没说过的话、没发过的动作、表情、图片、抱抱、场景或其他任何内容。
# - 严禁生成或添加“来访者当前输入”“用户下一句”“额外对话”“后续对话”等虚构部分。
# - {context} 是外部知识，仅用于背景参考，严禁当成对话历史、用户输入或可续写的内容。
# - 严禁重复同一句话、段落或类似内容。
# - 回复必须完全基于用户真实输入 {query} 和真实历史 {history}，保持真实。
# - 用自然、像朋友一样的语气，结合用户情绪词进行共情和陪伴。
# - 字数严格控制在180-280字之间，只回复一次。
#
# 真实历史对话（仅此，不要改动或续写）：{history}
# 外部知识（仅参考背景，不要列出、引用或当成历史）：{context}
# 当前用户真实输入（仅此一句，不要添加任何额外输入）：{query}
#
# 请以第一人称、温暖语气回复，表达理解与陪伴。 [/INST]"""
#     ),
#
#     "2": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""[INST] 你是一个专业的心理咨询师。现在用户在寻求建议阶段，需要实用、可操作的指导。
#
# 【绝对严格规则，必须100%遵守】
# - 只输出一次纯建议文本，严禁输出任何系统提示、角色标签、示例、解释、推理过程、思考过程。
# - 严禁复述用户原话、改写用户输入、添加“用户说”“来访者说”等。
# - 严禁编造用户没说过的话、没发过的动作或内容。
# - 严禁生成或添加“来访者当前输入”“用户下一句”“额外对话”“后续对话”等虚构部分。
# - {context} 是外部知识，仅用于参考背景，严禁当成对话历史或用户输入。
# - 严禁重复同一句话、段落或类似内容。
# - 回复必须基于用户真实问题 {query} 和真实历史 {history}。
# - 用自然、专业语气，先简短共情，再给出清晰步骤化建议（建议用编号或分点）。
# - 字数严格控制在250-350字之间，只回复一次。
#
# 真实历史对话（仅此，不要改动或续写）：{history}
# 外部知识（仅参考背景，不要列出、引用或当成历史）：{context}
# 当前用户真实问题（仅此一句，不要添加任何额外输入）：{query}
#
# 请以第一人称、专业语气给出建议。 [/INST]"""
#     ),
#
#     "3": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""[INST] 你是一个高度警觉的危机干预咨询师。现在用户可能处于高风险状态，安全第一。
#
# 【绝对严格规则，必须100%遵守】
# - 只输出一次纯回复文本，严禁输出任何系统提示、角色标签、示例、解释、推理过程、思考过程。
# - 严禁复述用户原话、改写用户输入、添加“用户说”“来访者说”等。
# - 严禁编造用户没说过的话、没发过的动作或内容。
# - 严禁生成或添加“来访者当前输入”“用户下一句”“额外对话”“后续对话”等虚构部分。
# - {context} 是外部知识，仅用于参考背景，严禁当成对话历史或用户输入。
# - 严禁重复同一句话、段落或类似内容。
# - 回复必须基于用户真实表达 {query} 和真实历史 {history}。
# - 立即表达关心与重视，强烈建议寻求专业即时帮助（热线：北京心理危机干预热线 800-810-1117 / 010-82951332，或当地急救120）。
# - 字数严格控制在180-280字之间，只回复一次。
#
# 真实历史对话（仅此，不要改动或续写）：{history}
# 外部知识（仅参考背景，不要列出、引用或当成历史）：{context}
# 当前用户真实表达（仅此一句，不要添加任何额外输入）：{query}
#
# 请以第一人称、关怀语气回复，立即引导求助。 [/INST]"""
#     )
# }
#
#
# # ================= Agent2 =================
# def agent2_generate(intent: str, query: str) -> str:
#     # 1️⃣ 意图合法性检查
#     if intent not in prompts:
#         return "抱歉，我暂时无法理解您的意图。"
#
#     # 2️⃣ 读取最近4轮历史
#     history_msgs = memory.chat_memory.messages[-4:]
#     history_str = "\n".join(
#         [f"{msg.type}: {msg.content}" for msg in history_msgs]
#     )
#
#     # 3️⃣ RAG 检索
#     docs = retriever.get_relevant_documents(query)
#     context = "\n\n".join(
#         [doc.page_content[:500] for doc in docs]
#     )
#
#     # 4️⃣ 构建链
#     chain = LLMChain(
#         llm=llm,
#         prompt=prompts[intent]
#     )
#
#     # 5️⃣ 调用模型
#     response = chain.invoke({
#         "history": history_str,
#         "query": query,
#         "context": context
#     })
#
#     # 6️⃣ 取出文本（invoke返回dict）
#     if isinstance(response, dict) and "text" in response:
#         response_text = response["text"]
#     else:
#         response_text = str(response)
#
#     # 7️⃣ 保存记忆
#     memory.save_context(
#         {"input": query},
#         {"output": response_text}
#     )
#
#     return response_text.strip()
#
# # ================= 总 pipeline =================
# def consultation_pipeline(user_query: str) -> str:
#     history_msgs = memory.chat_memory.messages[-4:]
#     history_str = "\n".join([msg.content for msg in history_msgs])
#
#     intent = get_intent(history=history_str, query=user_query, examples=GLOBAL_EXAMPLES)
#     print(f"[路由] 检测到意图: {intent}")
#
#     reply = agent2_generate(intent, user_query)
#     return reply
#
# # ================= 测试 =================
# if __name__ == "__main__":
#     print("=== 多轮心理咨询测试（输入 quit 退出） ===")
#
#     while True:
#         user_input = input("你: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             break
#
#         reply = consultation_pipeline(user_input)
#         print(f"AI: {reply}\n")
#         print("-" * 80)

import os
import torch
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from agent1_intent_classifier import get_intent, load_and_sample_examples

# ================= 显存优化 =================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# ================= 路径配置 =================
PROJECT_ROOT = "/tmp/pycharm_project_769/MentalHealth-Counselling-System"
BASE_MISTRAL_PATH = "/autodl-fs/data/models/Mistral-7B-Instruct-v0.3"
FINETUNED_ADAPTER_PATH = "/autodl-fs/data/models/tuned_model"
EMBED_PATH = "/autodl-fs/data/models/llama-embed-nemotron-8b"
FAISS_INDEX_PATH = "/tmp/pycharm_project_769/MentalHealth-Counselling-System/baseline_v2_llama/faiss_index_llama"
DATA_FILE = os.path.join(PROJECT_ROOT, "1040data_full_labeled_ALL.jsonl")
torch_dtype = torch.float16

# ================= tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(BASE_MISTRAL_PATH, trust_remote_code=True)

# 预加载 Few-shot 示例
try:
    print(f"正在从 {DATA_FILE} 加载意图分类示例...")
    GLOBAL_EXAMPLES = load_and_sample_examples(DATA_FILE, num_per_class=4)
except Exception as e:
    print(f"警告：加载示例失败，将使用兜底逻辑。错误: {e}")
    GLOBAL_EXAMPLES = []

# ================= 生成模型 =================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MISTRAL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

base_model.config.use_cache = False

model = PeftModel.from_pretrained(
    base_model,
    FINETUNED_ADAPTER_PATH,
    torch_dtype=torch_dtype,
    is_trainable=False
)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=384,
    temperature=0.7,
    do_sample=False,
    return_full_text=False,
)

# llm_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=200,  # 更短，强制早停
#     temperature=0.01,    # 几乎确定性
#     top_p=0.7,
#     do_sample=False,
#     repetition_penalty=2.0,  # 极强惩罚重复
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.pad_token_id,
#     return_full_text=False,
# )

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

# ================= 内存（升级到新接口） =================
chat_history = ChatMessageHistory()

# ================= Prompt =================
# prompts = {
#     "1": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""[INST] 你是一个温暖、共情的心理倾听者。现在用户在倾诉情绪，只需要被理解和陪伴。
#
# 全程用中文回复。
# 输出必须：
# - 只输出纯回复文本，什么都不要加。
# - 禁止出现“用户当前输入”“来访者”“如果”“原则”“我们可以尝试”“你愿意试试”“认知行为疗法”等任何词。
# - 禁止复述、重复用户原话。
# - 禁止编造用户没说过的话、动作、场景、治疗建议。
# - 禁止生成任何形式的后续对话、下一句、示例。
# - {context} 完全禁止在输出中出现、提及、列出或引用。
# - 禁止重复任何句子或段落。
# - 回复必须完全基于真实输入 {query} 和真实历史 {history}。
# - 用自然、温暖、像朋友一样的中文语气表达共情和陪伴。
# - 字数 150-220 字，只回复一次。
#
# 真实历史：{history}
# 用户输入：{query} [/INST]"""
#     ),
#
#     "2": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""[INST] 你是一个专业的心理咨询师。现在用户在寻求建议阶段，需要实用指导。
#
# 全程用中文回复。
# 输出必须：
# - 只输出纯建议文本，什么都不要加。
# - 禁止出现“用户当前输入”“来访者”“如果”“原则”“我们可以尝试”“你愿意试试”“认知行为疗法”等任何词。
# - 禁止复述、重复用户原话。
# - 禁止编造用户没说过的话、动作、场景、治疗建议。
# - 禁止生成任何形式的后续对话、下一句、示例。
# - {context} 完全禁止在输出中出现、提及、列出或引用。
# - 禁止重复任何句子或段落。
# - 回复必须基于真实问题 {query} 和真实历史 {history}。
# - 用自然、专业中文语气，先简短共情，再给出清晰步骤化建议（用编号）。
# - 字数 220-320 字，只回复一次。
#
# 真实历史：{history}
# 用户问题：{query} [/INST]"""
#     ),
#
#     "3": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""[INST] 你是一个高度警觉的危机干预咨询师。现在用户可能处于高风险状态，安全第一。
#
# 全程用中文回复。
# 输出必须：
# - 只输出纯回复文本，什么都不要加。
# - 禁止出现“用户当前输入”“来访者”“如果”“原则”“我们可以尝试”“你愿意试试”“认知行为疗法”等任何词。
# - 禁止复述、重复用户原话。
# - 禁止编造用户没说过的话、动作、场景。
# - 禁止生成任何形式的后续对话、下一句、示例。
# - {context} 完全禁止在输出中出现、提及、列出或引用。
# - 禁止重复任何句子或段落。
# - 立即表达关心，强烈建议拨打专业热线（北京心理危机干预热线 800-810-1117 / 010-82951332，或当地急救120）。
# - 字数 150-220 字，只回复一次。
#
# 真实历史：{history}
# 用户表达：{query} [/INST]"""
#     )
# }

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
- 字数 150-220 字，只回复一次。

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
- 字数 220-320 字，只回复一次。

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
- 字数 150-220 字，只回复一次。

真实历史：{history}
用户表达：{query}

回复： [/INST]"""
    )
}

# prompts = {
#     "1": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""[INST] 你是一个温暖、共情的心理倾听者。现在用户在倾诉情绪，只需要被理解和陪伴。
#
# 输出必须：
# - 只输出纯回复文本，什么都不要加。
# - 禁止出现“用户当前输入”“来访者”“如果”“原则”“我们可以尝试”“你愿意试试”“认知行为疗法”等任何词。
# - 禁止复述、重复用户原话。
# - 禁止编造用户没说过的话、动作、场景、治疗建议。
# - 禁止生成任何形式的后续对话、下一句、示例。
# - {context} 完全禁止在输出中出现、提及、列出或引用。
# - 禁止重复任何句子或段落。
# - 回复必须完全基于真实输入 {query} 和真实历史 {history}。
# - 用自然、温暖语气表达共情和陪伴。
# - 字数 150-220 字，只回复一次。
#
# 真实历史：{history}
# 用户输入：{query} [/INST]"""
#     ),
#
#     "2": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""[INST] 你是一个专业的心理咨询师。现在用户在寻求建议阶段，需要实用指导。
#
# 输出必须：
# - 只输出纯建议文本，什么都不要加。
# - 禁止出现“用户当前输入”“来访者”“如果”“原则”“我们可以尝试”“你愿意试试”“认知行为疗法”等任何词。
# - 禁止复述、重复用户原话。
# - 禁止编造用户没说过的话、动作、场景、治疗建议。
# - 禁止生成任何形式的后续对话、下一句、示例。
# - {context} 完全禁止在输出中出现、提及、列出或引用。
# - 禁止重复任何句子或段落。
# - 回复必须基于真实问题 {query} 和真实历史 {history}。
# - 用自然、专业语气，先简短共情，再给出清晰步骤化建议（用编号）。
# - 字数 220-320 字，只回复一次。
#
# 真实历史：{history}
# 用户问题：{query} [/INST]"""
#     ),
#
#     "3": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""[INST] 你是一个高度警觉的危机干预咨询师。现在用户可能处于高风险状态，安全第一。
#
# 输出必须：
# - 只输出纯回复文本，什么都不要加。
# - 禁止出现“用户当前输入”“来访者”“如果”“原则”“我们可以尝试”“你愿意试试”“认知行为疗法”等任何词。
# - 禁止复述、重复用户原话。
# - 禁止编造用户没说过的话、动作、场景。
# - 禁止生成任何形式的后续对话、下一句、示例。
# - {context} 完全禁止在输出中出现、提及、列出或引用。
# - 禁止重复任何句子或段落。
# - 立即表达关心，强烈建议拨打专业热线（北京心理危机干预热线 800-810-1117 / 010-82951332，或当地急救120）。
# - 字数 150-220 字，只回复一次。
#
# 真实历史：{history}
# 用户表达：{query} [/INST]"""
#     )
# }

# ================= Agent2 生成函数（升级 RunnableSequence） =================
def agent2_generate(intent: str, query: str) -> str:
    if intent not in prompts:
        return "抱歉，我暂时无法理解您的意图。"

    # 取最近4轮历史
    history_msgs = chat_history.messages[-4:]
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])

    # RAG 检索（改用 invoke）
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content[:500] for doc in docs])

    # 构建新链（RunnableSequence）
    chain = (
        {"history": RunnablePassthrough(), "query": RunnablePassthrough(), "context": RunnablePassthrough()}
        | prompts[intent]
        | llm
        | StrOutputParser()
    )

    # 加历史管理
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,
        input_messages_key="query",
        history_messages_key="history",
    )

    # 调用
    response = chain_with_history.invoke(
        {"query": query, "context": context},
        config={"configurable": {"session_id": "default"}}
    )

    # 保存历史（自动由 RunnableWithMessageHistory 处理，但手动加保险）
    chat_history.add_user_message(query)
    chat_history.add_ai_message(response)

    # return response.strip()
    response = response.strip()

    # 黑名单截断（任何出现就砍掉后面）
    black_list = [
        "用户当前输入", "来访者当前输入", "回复原则", "如果用户", "我们可以通过"
    ]
    for phrase in black_list:
        if phrase in response:
            response = response.split(phrase)[0].strip()

    # 去掉复述开头
    if response.startswith(query):
        response = response[len(query):].lstrip('，。！？ \n').strip()

    # 强制截断在句号
    if len(response) > 300:
        response = response[:300].rsplit('。', 1)[0] + '。'

    return response

# ================= 总 pipeline =================
def consultation_pipeline(user_query: str) -> str:
    # 历史字符串（从 chat_history 取）
    history_msgs = chat_history.messages[-4:]
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])

    intent = get_intent(history=history_str, query=user_query, examples=GLOBAL_EXAMPLES)
    print(f"[路由] 检测到意图: {intent}")

    reply = agent2_generate(intent, user_query)
    return reply

# ================= 测试 =================
if __name__ == "__main__":
    print("=== 多轮心理咨询测试（输入 quit 退出） ===")

    while True:
        user_input = input("你: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        reply = consultation_pipeline(user_input)
        print(f"AI: {reply}\n")
        print("-" * 80)
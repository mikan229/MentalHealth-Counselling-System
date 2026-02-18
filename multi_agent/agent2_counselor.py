#
#
# import os
# # 强制开启离线模式，避免联网检查导致版本冲突或卡顿
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# # 优化显存分配，减少碎片
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
# from langchain_core.prompts import PromptTemplate
# from langchain_core.documents import Document
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import LLMChain
# from langchain_huggingface import HuggingFacePipeline
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
#
# # ================= 配置区（全部本地路径） =================
# # GPU 设置
# device_0 = "cuda:0"
# device_1 = "cuda:1"
# torch_dtype = torch.float16
#
# # 1. 生成模型：Mistral-7B-Instruct-v0.3
# mistral_local_path = "/autodl-fs/data/models/Mistral-7B-Instruct-v0.3"
#
# # 生成模型量化配置
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )
#
# tokenizer = AutoTokenizer.from_pretrained(mistral_local_path)
# model = AutoModelForCausalLM.from_pretrained(
#     mistral_local_path,
#     quantization_config=quant_config,
#     device_map=device_0, # 强制锁定在 GPU 0
#     dtype=torch_dtype,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# )
#
# # 包装为 LangChain 可用的 LLM 对象
# raw_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     temperature=0.7,
#     do_sample=True,
# )
# llm = HuggingFacePipeline(pipeline=raw_pipeline)
#
# # 2. 嵌入模型：llama-embed-nemotron-8b
# embed_model_local_path = "/autodl-fs/data/models/llama-embed-nemotron-8b"
#
# # 为巨大的 8B 嵌入模型开启 4-bit 量化以防 GPU 1 再次 OOM
# embed_quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )
#
# embeddings = HuggingFaceEmbeddings(
#     model_name=embed_model_local_path,
#     model_kwargs={
#         "local_files_only": True,
#         "device": device_1,  # 强制锁定在 GPU 1
#         "trust_remote_code": True,
#         "model_kwargs": {"quantization_config": embed_quant_config} # 嵌入模型量化
#     },
#     encode_kwargs={"normalize_embeddings": True}
# )
#
# # 3. 加载本地 FAISS 索引
# faiss_folder = "/tmp/pycharm_project_337/MentalHealth-Counselling-System/baseline_v2_llama/faiss_index_llama"
# db = FAISS.load_local(
#     folder_path=faiss_folder,
#     embeddings=embeddings,
#     allow_dangerous_deserialization=True
# )
#
# retriever = db.as_retriever(search_kwargs={"k": 4})
#
# # 4. 对话内存
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#
# # ================= 意图专属 Prompt（保持不变） =================
# prompts = {
#     "1": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""你是一个温暖、共情的心理倾听者。现在用户在情感表达阶段，主要需要被理解和陪伴。
# 历史对话：{history}
# 当前用户说：{query}
# 检索知识：{context}
# 你的回复："""
#     ),
#     "2": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""你是一个专业的心理咨询师。现在用户在寻求建议阶段，需要实用、可操作的指导。
# 历史对话：{history}
# 当前用户问题：{query}
# 检索知识：{context}
# 你的建议："""
#     ),
#     "3": PromptTemplate(
#         input_variables=["history", "query", "context"],
#         template="""你是一个高度警觉的危机干预咨询师。现在用户可能处于危机求助阶段，安全第一。
# 历史对话：{history}
# 当前用户表达：{query}
# 检索知识：{context}
# 你的紧急回复："""
#     )
# }
#
# # ================= Agent2 生成函数 =================
# def agent2_generate(intent: str, query: str) -> str:
#     if intent not in prompts:
#         return "抱歉，我暂时无法理解您的意图。请再描述一下您的感受或需求。"
#
#     # 获取历史
#     history_msgs = memory.chat_memory.messages[-6:]
#     history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])
#
#     # RAG 检索
#     docs = retriever.invoke(query)
#     context = "\n\n".join([doc.page_content for doc in docs])
#
#     # 生成：使用包装后的 llm 对象
#     chain = LLMChain(llm=llm, prompt=prompts[intent])
#     response = chain.run(
#         history=history_str,
#         query=query,
#         context=context
#     )
#
#     # 保存内存
#     memory.save_context({"input": query}, {"output": response})
#     return response.strip()
#
# # ================= 从 Agent1 导入 get_intent =================
# # 确保该文件在同一目录下
# try:
#     from agent1_intent_classifier import get_intent
# except ImportError:
#     def get_intent(history, query): return "1" # 临时兜底
#
# # ================= 完整 Pipeline =================
# def consultation_pipeline(user_query: str) -> str:
#     history_msgs = memory.chat_memory.messages[-5:]
#     history_str = "\n".join([msg.content for msg in history_msgs])
#
#     intent = get_intent(history=history_str, query=user_query)
#     print(f"[路由] 检测到意图: {intent}")
#
#     reply = agent2_generate(intent, user_query)
#     return reply
#
# # ================= 交互测试 =================
# if __name__ == "__main__":
#     print("=== 多轮心理咨询对话测试（输入 'quit' 退出） ===\n")
#     while True:
#         user_input = input("你: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             break
#         reply = consultation_pipeline(user_input)
#         print(f"AI: {reply}\n" + "-"*80)

import os
import torch

# 1. 强制环境优化
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 导入 Agent 1 的核心功能
from agent1_intent_classifier import get_intent, load_and_sample_examples

# ================= 配置区（全部本地路径） =================
# 路径修正：请确保这里的 337 与你当前的 AutoDL 路径一致
PROJECT_ROOT = "/tmp/pycharm_project_337/MentalHealth-Counselling-System"
DATA_FILE = os.path.join(PROJECT_ROOT, "1040data_full_labeled_ALL.jsonl")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "baseline_v2_llama/faiss_index_llama")
MISTRAL_PATH = "/autodl-fs/data/models/Mistral-7B-Instruct-v0.3"
EMBED_PATH = "/autodl-fs/data/models/llama-embed-nemotron-8b"

# 2. 预加载 Few-shot 示例 (修复 ValueError)
try:
    print(f"正在从 {DATA_FILE} 加载意图分类示例...")
    GLOBAL_EXAMPLES = load_and_sample_examples(DATA_FILE, num_per_class=4)
except Exception as e:
    print(f"警告：加载示例失败，将使用兜底逻辑。错误: {e}")
    GLOBAL_EXAMPLES = []

# 3. 加载生成模型 (锁定 GPU 0)
print("正在加载生成模型 Mistral-7B (GPU 0)...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MISTRAL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MISTRAL_PATH,
    quantization_config=quant_config,
    device_map="cuda:0",  # 强制单卡分配
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.7,
                        do_sample=True)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 4. 加载嵌入模型 (锁定 GPU 1 + 4-bit 量化)
print("正在加载嵌入模型 Nemotron-8B (GPU 1)...")
embed_quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_PATH,
    model_kwargs={
        "local_files_only": True,
        "device": "cuda:1",
        "trust_remote_code": True,
        "model_kwargs": {"quantization_config": embed_quant_config}
    },
    encode_kwargs={"normalize_embeddings": True}
)

# 5. 加载 FAISS
print("正在读取 FAISS 索引...")
db = FAISS.load_local(folder_path=FAISS_INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

# 6. 对话内存
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ================= 优化后的意图专属 Prompt =================
# 使用 Mistral 指令格式 [INST] ... [/INST] 防止模型复读系统指令
prompts = {
    "1": PromptTemplate(
        input_variables=["history", "query", "context"],
        template="""[INST] 你是一个温暖、共情的心理倾听者。
当前用户处于“情感表达”阶段，他们需要的是理解、接纳和陪伴，而不是说教或立即解决问题。

回复要求：
1. 语气必须柔和、像真人，严禁使用“1. 2. 3.”这种列表格式。
2. 严禁复读本指令内容。
3. 结合对话历史和检索到的知识，给出共情回应。

历史对话：{history}
检索知识（仅参考）：{context}
当前用户输入：{query} [/INST]
你的温暖回复："""
    ),
    "2": PromptTemplate(
        input_variables=["history", "query", "context"],
        template="""[INST] 你是一个专业的心理咨询师。
当前用户正在“寻求建议”，他们需要科学、实用且具有支持性的指导。

回复要求：
1. 先简短肯定用户的努力。
2. 结合检索到的心理学专业知识给出建议。
3. 建议要具体、可操作。

历史对话：{history}
检索知识：{context}
当前用户输入：{query} [/INST]
你的专业建议："""
    ),
    "3": PromptTemplate(
        input_variables=["history", "query", "context"],
        template="""[INST] 你是一个冷静、资深的危机干预专家。
当前用户表达了“自杀或自残”倾向，处于极高风险状态。

回复要求：
1. 立即表达对用户生命的重视，语气要坚定且温暖。
2. 必须引导用户寻求专业医疗机构或拨打心理危机干预热线（如：010-82951332）。
3. 保持简短，不要给复杂的建议，核心是安全。

历史对话：{history}
检索知识：{context}
当前用户输入：{query} [/INST]
你的紧急回复："""
    )
}


# ================= 业务逻辑函数 =================
def agent2_generate(intent: str, query: str) -> str:
    if intent not in prompts:
        return "抱歉，我暂时无法理解您的意图。"

    # 1. 准备输入
    history_msgs = memory.chat_memory.messages[-6:]
    history_str = "\n".join([f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in history_msgs])
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. 构建完整的 Prompt
    full_prompt = prompts[intent].format(history=history_str, query=query, context=context)

    # 3. 关键修正：手动截断输入部分
    # 我们不使用 chain.run，而是直接用 llm 接口或手动处理输出
    raw_output = llm.invoke(full_prompt)

    # 如果 llm.invoke 还是返回全文，使用以下逻辑：
    if full_prompt in raw_output:
        response = raw_output.replace(full_prompt, "").strip()
    else:
        # 兼容性处理：如果没找到全文，尝试只取最后一部分
        response = raw_output.split("你的温暖回复：")[-1].split("你的紧急回复：")[-1].strip()

    # 4. 存入内存（只存 AI 说的话，不存 Prompt！）
    memory.save_context({"input": query}, {"output": response})

    return response


def consultation_pipeline(user_query: str) -> str:
    # 提取历史用于意图识别
    history_msgs = memory.chat_memory.messages[-5:]
    history_str = "\n".join([msg.content for msg in history_msgs])

    # Agent 1 路由逻辑 (传入预加载的 GLOBAL_EXAMPLES)
    intent = get_intent(history=history_str, query=user_query, examples=GLOBAL_EXAMPLES)
    print(f"[智能体路由] 检测到意图: {intent}")

    # Agent 2 生成逻辑
    return agent2_generate(intent, user_query)


# ================= 主程序交互 =================
if __name__ == "__main__":
    print("\n" + "=" * 30)
    print("心理健康咨询系统 (多智能体版) 已就绪")
    print("=" * 30 + "\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        try:
            reply = consultation_pipeline(user_input)
            print(f"\nAI: {reply}\n")
        except Exception as e:
            print(f"\n系统运行出错: {e}")
        print("-" * 80)
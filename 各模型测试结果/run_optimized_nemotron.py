import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ================= 显存与离线配置 =================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
torch.cuda.empty_cache()

# ================= 精准修正的路径 =================
BASE_MISTRAL_PATH = "/autodl-fs/data/models/Mistral-7B-Instruct-v0.3"
EMBED_PATH = "/autodl-fs/data/models/llama-embed-nemotron-8b"
FAISS_INDEX_PATH = "/tmp/pycharm_project_769/MentalHealth-Counselling-System/baseline_v2_llama/faiss_index_llama"

print("==================================================")
print("[1/3] Loading 15GB Nemotron retriever...")
# 💡 核心修正：将 trust_remote_code 显式地传入 encode_kwargs 或作为独立参数，确保底层库不报错
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_PATH,
    model_kwargs={
        'device': 'cuda:0',
        'trust_remote_code': True  # 保留在这里以防万一
    },
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 8}  # 稍微调小 batch_size，给生成的 Mistral 留足显存
)

print("[2/3] Loading FAISS index...")
vectorstore = FAISS.load_local(
    folder_path=FAISS_INDEX_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
# 根据队友最新优化，召回前 3 条知识
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("[3/3] Loading Mistral with Native PyTorch...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MISTRAL_PATH, trust_remote_code=True)
# 修复 padding 问题，避免生成时产生杂音
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MISTRAL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.eval()
model.config.use_cache = False

print("\nReady! Starting 200 questions...")
print("==================================================")

input_excel = "/root/autodl-tmp/200测试结果集_全模型版.xlsx"
output_excel = "/root/autodl-tmp/200测试结果集_全模型版_优化后.xlsx"

try:
    df = pd.read_excel(input_excel)
except Exception as e:
    print(f"❌ 读取 Excel 失败: {e}")
    exit()

new_col_name = "优化后_RAG_Nemotron回答"
df[new_col_name] = ""

for index, row in tqdm(df.iterrows(), total=len(df)):
    history_text = str(row.get("历史对话", ""))
    if history_text.strip() == "" or history_text.lower() == "nan":
        history_text = "无"
    question = str(row.get("模拟用户提问", ""))

    # 1. 检索
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    # 2. 构建 Prompt
    prompt = f"<s>[INST] 你是一位心理咨询师。请基于以下权威知识和对话历史回答。\n\n知识参考：\n{context}\n\n对话历史：\n{history_text}\n\n当前问题：\n{question}\n\n回答要求：\n- 只用中文回答。\n- 结合历史对话，给予情感支持，避免直接诊断。\n[/INST]"

    # 3. 推理 (加上 attention_mask 避免警告)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=384,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. 解析
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response = response.replace("YOU", "你").replace("everyone", "每个人").strip()

    df.at[index, new_col_name] = response

# 保存
df.to_excel(output_excel, index=False)
print(f"🏆 Done! Saved to: {output_excel}")
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
torch.cuda.empty_cache()

BASE_MISTRAL_PATH = "/root/.cache/huggingface/hub/models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit/snapshots/d5f623888f1415cf89b5c208d09cb620694618ee"
EMBED_PATH = "/root/autodl-tmp/llama-embed-nemotron-8b"
FAISS_INDEX_PATH = "/root/autodl-tmp/faiss_index_llama/faiss_index_llama"

print("==================================================")
print("[1/3] 正在加载 15GB 的 Nemotron 检索器...")
embeddings = HuggingFaceEmbeddings(
model_name=EMBED_PATH,
model_kwargs={'device': 'cuda:0', 'trust_remote_code': True},
encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
)

vectorstore = FAISS.load_local(
folder_path=FAISS_INDEX_PATH,
embeddings=embeddings,
allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print("[2/3] 正在加载纯净版 Mistral 生成模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MISTRAL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
BASE_MISTRAL_PATH,
device_map="cuda:0",
torch_dtype=torch.bfloat16,
low_cpu_mem_usage=True,
trust_remote_code=True
)

llm_pipeline = pipeline(
"text-generation",
model=model,
tokenizer=tokenizer,
max_new_tokens=512,
temperature=0.7,
do_sample=True,
top_p=0.9,
return_full_text=False,
)

print("[3/3] 全部加载完毕！准备开启最后 100 题跑分...")
print("==================================================")

input_excel = "/root/autodl-tmp/100题_CPsyCoun半路拦截测试集.xlsx"
output_excel = "/root/autodl-tmp/RAG_Nemotron_多轮测试结果.xlsx"

try:
df = pd.read_excel(input_excel)
except Exception as e:
print(f"读取表格失败: {e}")
exit()

new_col_name = "RAG_Nemotron多轮回答"
df[new_col_name] = ""

for index, row in tqdm(df.iterrows(), total=len(df)):
history_text = str(row.get("历史对话 (History)", ""))
question = str(row.get("模拟用户提问", ""))

df.to_excel(output_excel, index=False)
print(f"大满贯达成！结果已保存至：{output_excel}")
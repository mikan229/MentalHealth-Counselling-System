# rag_setup_llama.py（PyCharm 编辑，AutoDL GPU 运行）

import os
import torch
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==================== 配置（AutoDL 路径） ====================
knowledge_folder = "/tmp/pycharm_project_427/MentalHealth-Counselling-System/psychology_knowledge/"
model_local_path  = "/autodl-fs/data/models/llama-embed-nemotron-8b"     # AutoDL 上模型
save_path         = "/root/faiss_index_llama"             # 索引先保存在 AutoDL

print("开始加载知识库...")
pdf_loader = DirectoryLoader(knowledge_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
pdf_docs = pdf_loader.load()
print(f"加载 PDF: {len(pdf_docs)} 个")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(pdf_docs)
print(f"切片后 chunk 数: {len(split_docs)}")

print("\n加载 Llama 嵌入模型（GPU）...")
embeddings = HuggingFaceEmbeddings(
    model_name=model_local_path,
    model_kwargs={
        'device': 'cuda',                    # AutoDL GPU
        'trust_remote_code': True
    },
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 32                     # GPU
    }
)

print("\n构建 FAISS 索引...")
vectorstore = FAISS.from_documents(split_docs, embeddings)
vectorstore.save_local(save_path)
print(f"索引保存到: {save_path}")

# 测试检索
print("\n=== 中文查询测试 ===")
queries = [
    "抑郁症的诊断标准是什么",
    "广泛性焦虑障碍的症状有哪些",
    "PTSD 的核心诊断特征"
]

for q in queries:
    print(f"\n查询: {q}")
    results = vectorstore.similarity_search(q, k=3)
    for i, doc in enumerate(results, 1):
        print(f"结果 {i}: {doc.page_content[:280]}...")
        print("来源:", doc.metadata.get("source", "未知"))
        print("-" * 80)
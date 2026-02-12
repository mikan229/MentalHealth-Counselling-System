# 导入必要的库
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,              # 用于 PDF，APA 官方 DSM-5-TR Fact Sheets（多个 PDF，覆盖常见精神障碍的诊断标准、症状、患病率、鉴别诊断等）
    TextLoader,               # 用于 TXT
    UnstructuredExcelLoader   # 用于 XLSX，WHO 的 ICD-11 MMS 精神与行为障碍章节（.txt 和 .xlsx 格式，提供结构化的国际诊断分类）
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# 包含心里资料的文件夹
folder_path = "psychology_knowledge/"  # 包含 PDF、TXT、XLSX 的文件夹

# 步骤1: 分别加载不同类型文件
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"文件夹未找到: {folder_path}")

# 加载 PDF
pdf_loader = DirectoryLoader(
    folder_path,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
pdf_docs = pdf_loader.load()
print(f"加载了 {len(pdf_docs)} 个 PDF 文档对象（来自 DSM Fact Sheets 等）。")

# 加载 TXT
txt_loader = DirectoryLoader(
    folder_path,
    glob="**/*.txt",
    loader_cls=lambda file_path: TextLoader(file_path, encoding="utf-8")
)
txt_docs = txt_loader.load()
print(f"加载了 {len(txt_docs)} 个 TXT 文档对象（来自 ICD-11 等）。")

# 加载 XLSX
excel_loader = DirectoryLoader(
    folder_path,
    glob="**/*.xlsx",
    loader_cls=UnstructuredExcelLoader
)
excel_docs = excel_loader.load()
print(f"加载了 {len(excel_docs)} 个 XLSX 文档对象（来自 ICD-11 表格等）。")

# 合并所有文档
all_documents = pdf_docs + txt_docs + excel_docs
print(f"总共加载了 {len(all_documents)} 个文档对象。")

# 步骤2: 文本切片
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每个 chunk 的最大字符数（可调整，根据内容密度）
    chunk_overlap=100,    # 重叠字符数，保持上下文连贯
    separators=["\n\n", "\n", " ", ""]  # 分隔符优先级
)
split_docs = text_splitter.split_documents(all_documents)
print(f"切片后得到 {len(split_docs)} 个 chunk。")

# 步骤3: 生成嵌入（向量化）
# 使用 Hugging Face 开源模型（第一次运行会自动下载）
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # 用 CPU；有 GPU 用 'cuda'
)

# embeddings = HuggingFaceEmbeddings(
#     model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
#     model_kwargs={'device': 'cuda'},
#     encode_kwargs={'normalize_embeddings': True},
#     trust_remote_code=True
# )

# 步骤4: 创建 FAISS 向量存储
vectorstore = FAISS.from_documents(
    documents=split_docs,
    embedding=embeddings
)
print("FAISS 索引创建完成。")

# 步骤5: 保存 FAISS 到本地
save_path = "faiss_index"  # 保存目录
vectorstore.save_local(save_path)
print(f"FAISS 索引已保存到 {save_path}。")

# 只用 DSM PDF 建一个独立的索引（作为主知识库）
dsm_docs = pdf_docs  # 只有 PDF
dsm_split = text_splitter.split_documents(dsm_docs)
dsm_vectorstore = FAISS.from_documents(dsm_split, embeddings)
dsm_vectorstore.save_local("faiss_index_dsm_only")
print("DSM 专用索引保存到 faiss_index_dsm_only")

# 原有索引作为补充

# 测试 DSM 专用索引
query = "抑郁症的诊断标准是什么"
dsm_results = dsm_vectorstore.similarity_search(query, k=3)
print("\n=== DSM 专用检索结果 ===")
for i, doc in enumerate(dsm_results, 1):
    print(f"结果 {i}: {doc.page_content[:300]}...")
    print("来源:", doc.metadata.get("source", "未知"))
    print("-" * 60)

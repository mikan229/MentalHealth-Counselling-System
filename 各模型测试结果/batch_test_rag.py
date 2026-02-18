import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from unsloth import FastLanguageModel
import torch

excel_file = "/mnt/d/liulanqi/MentalHealth-Counselling-System-main/MentalHealth-Counselling-System-main/心理咨询大模型基准测试集.xlsx"
output_file = "/mnt/d/liulanqi/MentalHealth-Counselling-System-main/MentalHealth-Counselling-System-main/RAG_Mistral测试结果.xlsx"
embed_model_path = "sentence-transformers/all-MiniLM-L6-v2"
index_path = "/mnt/d/liulanqi/MentalHealth-Counselling-System-main/MentalHealth-Counselling-System-main/faiss_index_dsm_only"

def post_process_answer(answer):
	answer = answer.replace("YOU", "你").replace("everyone", "每个人")
	return answer

print("\n[1/4] 正在加载翻译官 (向量嵌入模型)...")
embeddings = HuggingFaceEmbeddings(
	model_name=embed_model_path,
	model_kwargs={'device': 'cuda'},
	encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
)

print("\n[2/4] 正在加载心理学知识库 (FAISS)...")
vectorstore = FAISS.load_local(
	index_path,
	embeddings,
	allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("\n[3/4] 正在加载 Mistral 大模型 (Unsloth 4bit)...")
model, tokenizer = FastLanguageModel.from_pretrained(
	model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
	max_seq_length=4096,
	dtype=None,
	load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

print("\n[4/4] 开始读取 Excel 测试集...")
try:
	df = pd.read_excel(excel_file)
	df.columns = df.columns.str.strip()
except Exception as e:
	print(f"读取表格失败: {e}")
	exit()

question_col_name = "模拟用户提问"
if question_col_name not in df.columns:
	print(f"报错：找不到【{question_col_name}】列！")
	exit()

answers = []
print(f"成功读取！一共发现了 {len(df)} 道题。即将开始 RAG 自动化跑题...")

for index, row in df.iterrows():
	question = str(row[question_col_name])
	print(f"\n--- 正在思考第 {index + 1}/{len(df)} 题 ---")
	
	# 1. 检索知识库
	docs = retriever.invoke(question)
	context = "\n".join([doc.page_content for doc in docs])
	
	# 2. 拼接 Prompt
	prompt_text = f"你是一位专业、富有同情心的心理咨询师。请根据以下来自 DSM-5-TR 的权威知识，结合用户的问题，提供专业的心理建议：\n\n【权威知识参考】\n{context}\n\n回答要求：\n- 只用中文回答\n- 结合上面的知识给出建议，避免直接下诊断\n- 给予用户适当的情感支持\n"
	
	messages = [
		{"role": "system", "content": prompt_text},
		{"role": "user", "content": question}
	]
	
	# 3. 生成回答
	inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
	outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True, temperature=0.7)
	response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
	
	final_answer = post_process_answer(response.strip())
	answers.append(final_answer)

df['RAG_Mistral回答'] = answers
df.to_excel(output_file, index=False)

print(f"\n=======================================================")
print(f"🎉 跑分结束！所有题目的 RAG 回答已保存到：")
print(f"RAG_Mistral测试结果.xlsx")
print(f"=======================================================")
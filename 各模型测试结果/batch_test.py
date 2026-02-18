import pandas as pd
from unsloth import FastLanguageModel
import torch

excel_file = "/mnt/d/liulanqi/MentalHealth-Counselling-System-main/MentalHealth-Counselling-System-main/心理咨询大模型基准测试集.xlsx"
output_file = "/mnt/d/liulanqi/MentalHealth-Counselling-System-main/MentalHealth-Counselling-System-main/Agent2_Mistral基座测试结果.xlsx"

print("正在加载模型...")
base_model_path = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(model_name=base_model_path, max_seq_length=4096, dtype=None, load_in_4bit=True)

lora_path = "/mnt/d/liulanqi/MentalHealth-Counselling-System-main/MentalHealth-Counselling-System-main/agent2_finetuned_model"
model.load_adapter(lora_path)
FastLanguageModel.for_inference(model)

df = pd.read_excel(excel_file)
question_col_name = "模拟用户提问"
answers = []

print(f"成功读取！一共发现了 {len(df)} 道题。")

for index, row in df.iterrows():
	question = str(row[question_col_name])
	print(f"--- 正在思考第 {index + 1}/{len(df)} 题 ---")
	messages = [{"role": "user", "content": question}]
	inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
	outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True)
	response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
	answers.append(response)

df['Agent2_Mistral回答'] = answers
df.to_excel(output_file, index=False)
print("跑分结束！全部回答已保存到 Agent2_Mistral测试结果.xlsx")
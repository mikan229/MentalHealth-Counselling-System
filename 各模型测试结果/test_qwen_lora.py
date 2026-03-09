import os
import pandas as pd
from unsloth import FastLanguageModel


os.environ["HF_ENDPOINT"] = ""


excel_file = "/root/autodl-tmp/心理咨询大模型基准测试集.xlsx"
output_file = "/root/autodl-tmp/Qwen2_5_第二次微调结果.xlsx"

print("==================================================")
print("🚀 [1/3] 正在云端极速下载 Qwen2.5 基础模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
	model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
	max_seq_length=4096,
	load_in_4bit=True,
	dtype=None,
)

print("📚 [2/3] 正在挂载队友的“微调技能书” (LoRA)...")


model.load_adapter("xixi-lsf/second-qwen-finetuned")
FastLanguageModel.for_inference(model)

print("✅ [3/3] 模型合体完毕！准备开启 100 题跑分...")
print("==================================================")

try:
	df = pd.read_excel(excel_file)
	df.columns = df.columns.str.strip()
except Exception as e:
	print(f"❌ 读取表格失败: {e}")
	exit()

answers = []

for index, row in df.iterrows():
	question = str(row["模拟用户提问"])
	print(f"--- 正在思考第 {index + 1}/{len(df)} 题 ---")
	
	# 使用队友要求的 Qwen2.5 专属模板
	messages = [
		{"role": "system", "content": "你是一位专业、富有同情心的心理咨询师。请根据专业知识提供建议，安抚用户情绪。"},
		{"role": "user", "content": question}
	]
	
	inputs = tokenizer.apply_chat_template(
		messages,
		tokenize=True,
		add_generation_prompt=True,
		return_tensors="pt"
	).to("cuda")
	
	# 队友指定的生成参数：温度0.7，top_p=0.9
	outputs = model.generate(
		input_ids=inputs,
		max_new_tokens=512,
		use_cache=True,
		temperature=0.7,
		top_p=0.9,
		pad_token_id=tokenizer.eos_token_id
	)
	
	response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
	
	# 预防性清理队友提到的停止符
	response = response.replace("<|im_end|>", "").replace("<|im_start|>user", "").strip()
	answers.append(response)

df['Qwen微调版回答'] = answers
df.to_excel(output_file, index=False)
print(f"\n🎉 100 题全部跑完！结果已保存为：Qwen2_5_第二次微调结果.xlsx")
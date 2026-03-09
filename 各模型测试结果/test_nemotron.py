import time
from langchain_huggingface import HuggingFaceEmbeddings

已经帮你算好的 AutoDL 云端绝对路径
model_path = "/root/autodl-tmp/MentalHealth-Counselling-System-main/llama-embed-nemotron-8b"

print("==================================================")
print("🚀 正在 AutoDL 云端唤醒 15GB 的 Nemotron-8B 超大模型...")
print("💎 拥有 24G 显存的你，现在可以为所欲为了！")
print("==================================================")

try:
	start_time = time.time()
	embeddings = HuggingFaceEmbeddings(
		model_name=model_path,
		model_kwargs={'device': 'cuda', 'trust_remote_code': True},
		encode_kwargs={'normalize_embeddings': True}
	)
	end_time = time.time()
	print(f"\n🎉 奇迹发生！15GB 重型模型加载成功！耗时: {end_time - start_time:.2f} 秒")
	
	print("\n🧠 开始测试编码能力...")
	test_text = "我最近总是睡不着觉，感到非常焦虑，是不是抑郁了？"
	vector = embeddings.embed_query(test_text)
	
	print(f"\n✅ 成功将文字转换成了高维向量！")
	print(f"📏 向量维度是: {len(vector)} 维 (Nemotron的标志性高维度)")
	print(f"🔢 前 5 个数字是: {vector[:5]}")
except Exception as e:
	print(f"\n❌ 加载失败！可能是文件还没传完，或者路径不对。报错信息：")
	print(e)
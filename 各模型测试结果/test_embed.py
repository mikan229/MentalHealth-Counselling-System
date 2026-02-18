from langchain_huggingface import HuggingFaceEmbeddings


model_path = "/root/autodl-tmp/这里换成你复制的路径/llama-embed-nemotron-8b"

print("\n🚀 正在 AutoDL 云端唤醒 Nemotron-8B 超大嵌入模型...")
print("24G 显存已就位，毫无压力！")

try:
	embeddings = HuggingFaceEmbeddings(
		model_name=model_path,
		model_kwargs={'device': 'cuda', 'trust_remote_code': True},
		encode_kwargs={'normalize_embeddings': True}
	)
	print("\n🎉 奇迹发生了！15GB 重型模型秒级加载成功！开始测试编码...")
	
	test_text = "我最近总是睡不着觉，感到非常焦虑。"
	vector = embeddings.embed_query(test_text)
	
	print(f"✅ 成功将文字转换成了向量！")
	print(f"这个向量的超长维度是: {len(vector)} 维")
	print(f"前 5 个数字是: {vector[:5]}")
except Exception as e:
	print(f"\n❌ 加载失败了！真实报错信息如下：")
	print(e)
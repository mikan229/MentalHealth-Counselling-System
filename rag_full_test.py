# # rag_full_test.py
#
# import os
# from vllm import LLM, SamplingParams
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
#
# def main():
#
#     # ==================== 配置 ====================
#     embed_model_path = "/autodl-fs/data/models/all-MiniLM-L6-v2"  # 嵌入模型路径
#     index_path = "/tmp/pycharm_project_427/MentalHealth-Counselling-System/baseline_v1_allMiniLM/faiss_index_dsm_only"  # 索引路径
#
#     print("当前工作目录:", os.getcwd())
#     print("嵌入模型路径是否存在:", os.path.exists(embed_model_path))
#
#     # 加载嵌入模型
#     print("\n加载嵌入模型...")
#     embeddings = HuggingFaceEmbeddings(
#         model_name=embed_model_path,
#         model_kwargs={'device': 'cuda'},  # 使用 GPU
#         encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
#     )
#
#     # 加载 FAISS 索引
#     print("\n加载 FAISS 索引...")
#     vectorstore = FAISS.load_local(
#         index_path,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 增加检索文档数
#
#     # 加载 Mistral
#     print("\n正在加载 Mistral-7B-Instruct-v0.3（GPU）...")
#     llm = LLM(
#         model="/autodl-fs/data/models/Mistral-7B-Instruct-v0.3",
#         dtype="float16",
#         gpu_memory_utilization=0.9,
#         max_model_len=4096
#     )
#
#     def rag_answer(question):
#         # 1. 检索知识库
#         docs = retriever.invoke(question)
#         context = "\n".join([doc.page_content for doc in docs])
#
#         # 2. 构建更具个性化的 Prompt
#         prompt = f"""你是一位专业、温暖、共情的心理咨询师。请根据以下权威知识（来自 DSM-5-TR 和 ICD-11）以及用户的问题，提供一个个性化的建议：
#
# {context}
#
# 用户问题：{question}
#
# 回答要求：
# - 请只用中文回答
# - 提供个性化的情感支持和建议
# - 如果用户询问的是心理健康问题，避免过于模板化，结合具体背景给予帮助
# - 适当地提到寻求专业帮助的建议，但要温和且理解用户的感受
#
# 回答：
# """
#
#         # 3. 生成回答
#         sampling_params = SamplingParams(
#             temperature=0.8,  # 增加多样性
#             max_tokens=1024   # 增加最大生成长度
#         )
#         outputs = llm.generate([prompt], sampling_params)
#         return outputs[0].outputs[0].text.strip()
#
#     # ==================== 测试 ====================
#     test_questions = [
#         "我最近总是睡不着觉，脑子很乱，很焦虑，该怎么办？",
#         "我经常想起过去的不开心事，感觉很痛苦，是不是 PTSD？",
#         "抑郁的时候什么都提不起兴趣，怎么才能好起来？",
#         "我总是担心很多事情，控制不住自己，是广泛性焦虑障碍吗？"
#     ]
#
#     for q in test_questions:
#         print(f"\n用户问题：{q}")
#         print("咨询师回答：")
#         print(rag_answer(q))
#         print("=" * 100)
#
# # ⭐⭐ 关键就是这一行 ⭐⭐
# if __name__ == "__main__":
#     main()
#
#

# rag_full_test.py

import os
from vllm import LLM, SamplingParams
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# 后处理函数，用于清理生成的文本
def post_process_answer(answer):
    # 例如，替换掉不应出现的英文单词
    answer = answer.replace("YOU", "你").replace("everyone", "每个人")
    # 可以根据需要添加更多规则或修正
    return answer


def main():
    # ==================== 配置 ====================
    embed_model_path = "/autodl-fs/data/models/all-MiniLM-L6-v2"  # 嵌入模型路径
    index_path = "/tmp/pycharm_project_427/MentalHealth-Counselling-System/baseline_v1_allMiniLM/faiss_index_dsm_only"  # 索引路径

    print("当前工作目录:", os.getcwd())
    print("嵌入模型路径是否存在:", os.path.exists(embed_model_path))

    # 加载嵌入模型
    print("\n加载嵌入模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_path,
        model_kwargs={'device': 'cuda'},  # 使用 GPU
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
    )

    # 加载 FAISS 索引
    print("\n加载 FAISS 索引...")
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 增加检索文档数

    # 加载 Mistral
    print("\n正在加载 Mistral-7B-Instruct-v0.3（GPU）...")
    llm = LLM(
        model="/autodl-fs/data/models/Mistral-7B-Instruct-v0.3",
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=4096
    )

    def rag_answer(question):
        # 1. 检索知识库
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])

        # 2. 构建更具个性化的 Prompt
        prompt = f"""你是一位专业、富有同情心的心理咨询师，具备深厚的心理学背景。请根据以下来自 DSM-5-TR 和 ICD-11 的权威知识，结合用户的问题，提供专业的心理建议，并展现出理解与支持：

{context}

用户问题：{question}

回答要求：
- 只用中文回答
- 请使用准确、专业的心理学术语，同时以亲切、理解的语气提供帮助
- 给予用户适当的情感支持，但避免过度温馨化，保持专业
- 避免直接诊断，只提供建议和指导
- 如果适用，请建议寻求专业帮助，并用温和的语气提出
"""

        # 3. 生成回答
        sampling_params = SamplingParams(
            temperature=0.7,  # 稍低温度保持专业性
            max_tokens=1024  # 增加最大生成长度
        )
        outputs = llm.generate([prompt], sampling_params)

        # 后处理生成的文本
        return post_process_answer(outputs[0].outputs[0].text.strip())

    # ==================== 测试 ====================
    test_questions = [
        "我最近总是睡不着觉，脑子很乱，很焦虑，该怎么办？",
        "我经常想起过去的不开心事，感觉很痛苦，是不是 PTSD？",
        "抑郁的时候什么都提不起兴趣，怎么才能好起来？",
        "我总是担心很多事情，控制不住自己，是广泛性焦虑障碍吗？"
    ]

    for q in test_questions:
        print(f"\n用户问题：{q}")
        print("咨询师回答：")
        print(rag_answer(q))
        print("=" * 100)



if __name__ == "__main__":
    main()

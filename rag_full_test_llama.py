import os
import torch
from vllm import LLM, SamplingParams
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def post_process_answer(answer):
    answer = answer.replace("YOU", "你").replace("everyone", "每个人")
    return answer.strip()


def main():

    embed_model_path = "/autodl-fs/data/models/llama-embed-nemotron-8b"
    index_path = "/tmp/pycharm_project_179/MentalHealth-Counselling-System/baseline_v2_llama/faiss_index_llama"

    print("====== 第一阶段：加载嵌入模型到 GPU0 ======")

    # 🔥 强制 embedding 用 cuda:0
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_path,
        model_kwargs={
            'device': 'cuda:0',
            'trust_remote_code': True
        },
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 16
        }
    )

    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print("GPU0 当前显存：")
    print(torch.cuda.memory_summary(device=0, abbreviated=True))

    print("\n====== 第二阶段：加载生成模型到 GPU1 ======")

    # 🔥 关键操作：让 vLLM 只看到 GPU1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    llm = LLM(
        model="/autodl-fs/data/models/Mistral-7B-Instruct-v0.3",
        dtype="float16",
        tensor_parallel_size=1,   # ❗必须改成1
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enforce_eager=True
    )

    def rag_answer(question):
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])

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

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1024
        )

        outputs = llm.generate([prompt], sampling_params)
        return post_process_answer(outputs[0].outputs[0].text.strip())

    test_questions = [
        "我最近总是睡不着觉，脑子很乱，很焦虑，该怎么办？",
        "我经常想起过去的不开心事，感觉很痛苦，是不是 PTSD？",
    ]

    for q in test_questions:
        print("\n用户问题：", q)
        print("咨询师回答：")
        print(rag_answer(q))
        print("=" * 80)


if __name__ == "__main__":
    main()

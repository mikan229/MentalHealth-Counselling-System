import os
import pandas as pd
import random
from datasets import load_dataset

# 1. 挂载 HuggingFace 国内镜像，防止 AutoDL 连外网失败
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("📥 正在从 HuggingFace 下载 CPsyCoun 数据集 (大概需要一两分钟，请耐心等待)...")

try:
    # 加载数据集（它默认是 LLaMA-Factory 的微调格式）
    dataset = load_dataset("CAS-SIAT-XinHai/CPsyCoun", split="train")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit()

# 2. 转换数据，并只挑选真正的“多轮”对话（历史记录 >= 2轮）
all_data = list(dataset)
multi_turn_data = []

for d in all_data:
    # 筛选：有历史记录，且至少聊了两个来回
    if 'history' in d and isinstance(d['history'], list) and len(d['history']) >= 2:
        multi_turn_data.append(d)

print(f"✅ 成功加载数据！从总库中筛出了 {len(multi_turn_data)} 条深度多轮对话。")

# 3. 随机抽取 100 条
sample_size = min(100, len(multi_turn_data))
sampled_data = random.sample(multi_turn_data, sample_size)

# 4. 转换成你现有的 Excel 评测表格式
excel_rows = []
for i, item in enumerate(sampled_data):
    # 将 history (列表格式: [["用户", "AI"], ...]) 格式化为纯文本前情提要
    history_list = item.get('history', [])
    history_text = "【历史对话前情提要】\n"
    for turn in history_list:
        if len(turn) >= 2:
            history_text += f"用户: {turn[0]}\n咨询师: {turn[1]}\n\n"

    # 当前用户的最新一句话
    current_input = str(item.get('instruction', '')) + str(item.get('input', ''))

    excel_rows.append({
        "编号": i + 1,
        "历史对话 (History)": history_text.strip(),
        "模拟用户提问": current_input.strip(),
        "评测侧重点 (供 GPT-4 打分参考)": "测试多轮引导能力：请重点考察 AI 是否能结合长历史上下文进行共情。AI 绝对不能急于给出总结性建议，而应继续循循善诱，像真实咨询师一样探索用户的深层情绪。"
    })

# 5. 生成 Excel 并保存
output_path = "/root/autodl-tmp/100题_CPsyCoun多轮测试集.xlsx"
df = pd.DataFrame(excel_rows)
df.to_excel(output_path, index=False)

print(f"🎉 搞定！100条极品多轮测试数据已生成！")
print(f"文件位置: {output_path}")
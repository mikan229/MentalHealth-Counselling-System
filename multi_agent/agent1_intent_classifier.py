# agent1_intent_classifier.py
# 已适配你的 .jsonl 格式：字段为 "history_up_to_prev", "current_user_input", "user_intent"

import requests
import json
import random
from pathlib import Path
from typing import List, Dict
from collections import Counter

# ====================== 配置区 ======================
API_KEY = "sk-gtvxdnbkcergwvxpsldbxlmlszfbnrvfqvntyumzleqgjvmh"  # ← 替换
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"  # 确认或问A

DATA_FILE = r"D:\daima\MentalHealth-Counselling-System\MentalHealth-Counselling-System\1040data_full_labeled_ALL.jsonl"  # ← 你的路径

# 中文意图 → 数字映射（和A保持一致）
INTENT_MAP = {
    "情感表达": "1",
    "寻求建议": "2",
    "危机求助": "3",
    # 如果有其他变体，加在这里
}


# ====================== 加载 & 采样 few-shot 示例 ======================
def load_and_sample_examples(file_path: str, num_per_class: int = 4) -> List[Dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析失败: {e} → 跳过")

    print(f"总共加载了 {len(data)} 条有效数据")

    # 按意图（数字）分组
    grouped = {"1": [], "2": [], "3": []}
    for item in data:
        chinese_intent = item.get("user_intent", "").strip()
        num_intent = INTENT_MAP.get(chinese_intent)
        if num_intent:
            grouped[num_intent].append(item)

    # 采样并转换为统一格式
    examples = []
    for num_intent, items in grouped.items():
        if len(items) > 0:
            sampled = random.sample(items, min(num_per_class, len(items)))
            for s in sampled:
                examples.append({
                    "history": s.get("history_up_to_prev", ""),
                    "query": s.get("current_user_input", ""),
                    "intent": num_intent
                })

    print(f"采样了 {len(examples)} 条 few-shot 示例")
    print("意图分布:", Counter(ex["intent"] for ex in examples))

    if len(examples) == 0:
        print("警告：没有匹配到任何意图！请检查 INTENT_MAP 是否覆盖所有 user_intent 值")

    return examples


# ====================== Prompt 构建（不变） ======================
def build_prompt(history: str, query: str, examples: List[Dict]) -> str:
    prompt = """你是一个心理咨询意图分类器。
根据历史对话和当前用户查询，将意图严格分类为以下之一：
1. 情感表达（用户主要倾诉情绪、宣泄感受、分享痛苦）
2. 寻求建议（用户询问方法、策略、如何处理具体问题）
3. 危机求助（涉及自杀、自残、极端伤害、求救等高风险内容）

规则：
- 只输出一个数字：1 或 2 或 3，不要任何其他文字、解释、标点。
- 优先判断是否为危机（3），因为安全最重要。
- 如果用户主要在倾诉感受、表达痛苦、没有明确问“怎么办”“怎么做”“有什么方法”，则为 1（情感表达），即使后面有感谢或未来计划。
- 只有明确询问方法、策略、建议时才判为 2。
示例：
"""
    for ex in examples:
        h = ex["history"].replace("\n", " ").strip()[:250]
        q = ex["query"].replace("\n", " ").strip()[:150]
        prompt += f"历史: {h}\n查询: {q}\n意图: {ex['intent']}\n\n"

    prompt += f"""现在分类：
历史: {history.replace("\n", " ").strip()}
查询: {query.replace("\n", " ").strip()}
意图: """
    return prompt


# ====================== API 调用（不变） ======================
def call_deepseek_api(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.05,
        "max_tokens": 8,
        "top_p": 0.85
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        for char in content:
            if char in "123":
                return char
        print(f"警告：模型输出异常 -> {content}")
        return "未知"
    except Exception as e:
        print(f"API 调用失败: {e}")
        return "错误"


# ====================== 核心函数（不变） ======================
def get_intent(history: str = "", query: str = "", examples: List[Dict] = None) -> str:
    if examples is None:
        raise ValueError("必须提供 few-shot examples")
    prompt = build_prompt(history, query, examples)
    intent = call_deepseek_api(prompt)
    return intent


# ====================== 批量评估（字段已适配） ======================
def evaluate_on_dataset(file_path: str, examples: List[Dict], test_size: int = 100):
    path = Path(file_path)
    if not path.exists():
        print("测试文件不存在，跳过评估")
        return

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                continue

    if len(data) == 0:
        print("没有有效数据")
        return

    test_samples = random.sample(data, min(test_size, len(data)))

    correct = 0
    total = 0
    error_examples = []

    print(f"\n开始评估 {len(test_samples)} 条样本...")
    for i, item in enumerate(test_samples, 1):
        chinese_intent = item.get("user_intent", "").strip()
        true_intent = INTENT_MAP.get(chinese_intent, "未知")
        if true_intent == "未知":
            continue  # 跳过无映射的

        pred = get_intent(
            history=item.get("history_up_to_prev", ""),
            query=item.get("current_user_input", ""),
            examples=examples
        )
        if pred == true_intent:
            correct += 1
        else:
            error_examples.append({
                "true": true_intent,
                "pred": pred,
                "query": item.get("current_user_input", "")[:60]
            })
        total += 1

        if i % 20 == 0:
            print(f"已完成 {i}/{total} 条... 当前准确率: {correct / total:.2%} (有效样本)")

    accuracy = correct / total if total > 0 else 0
    print(f"\n评估完成！")
    print(f"准确率 (Accuracy): {accuracy:.4f} ({correct}/{total})")

    if error_examples:
        print("\n前5个错误样例：")
        for ex in error_examples[:5]:
            print(f"true: {ex['true']} | pred: {ex['pred']} | query: {ex['query']}")


# ====================== 主程序 ======================
if __name__ == "__main__":
    try:
        few_shot_examples = load_and_sample_examples(DATA_FILE, num_per_class=4)  # 每类4条，共12条左右
    except Exception as e:
        print(f"加载失败: {e}")
        few_shot_examples = []

    if not few_shot_examples:
        print("没有 few-shot 示例，无法继续")
        exit(1)

    # 单条测试
    print("\n=== 单条手动测试 ===")
    manual_tests = [
        ("用户之前说婚姻问题很困扰", "我很失望，也很痛苦，不知道该怎么办。", "1"),
        ("", "有什么方法可以缓解这种情绪？", "2"),
        ("用户连续表达绝望", "我觉得活着没意义了，想结束一切", "3")
    ]
    for h, q, exp in manual_tests:
        pred = get_intent(h, q, few_shot_examples)
        print(f"历史: {h[:40]}...")
        print(f"查询: {q}")
        print(f"预测: {pred}   预期: {exp}\n")

    # 批量评估
    print("\n=== 批量评估 ===")
    evaluate_on_dataset(DATA_FILE, few_shot_examples, test_size=120)  # 可调大
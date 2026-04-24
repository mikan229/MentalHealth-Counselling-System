# finetune_agent2.py
# 使用 Unsloth 微调 Agent2（意图自适应回复生成）
# 所有路径统一放在 D:\PythonProject5_psy_LLM 下

#微调mistral

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ====================== 统一路径设置 ======================
BASE_DIR = r"D:\PythonProject5_psy_LLM"

# Hugging Face 缓存目录（模型自动下载到这里）
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")

# 数据路径（你的分层数据）
TRAIN_FILE = os.path.join(BASE_DIR, "hf_datasets", "train.jsonl")
VAL_FILE   = os.path.join(BASE_DIR, "hf_datasets", "val.jsonl")
TEST_FILE  = os.path.join(BASE_DIR, "hf_datasets", "test.jsonl")

# 微调后 LoRA 模型保存目录
OUTPUT_DIR = os.path.join(BASE_DIR, "agent2_finetuned_model")

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 模型参数 ======================
max_seq_length = 4096
dtype = None  # None = 自动检测
load_in_4bit = True

model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"  # 推荐模型

# ====================== 加载模型 ======================
print("加载模型（首次会自动下载到 hf_cache）...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 启用 LoRA
model = FastLanguageModel.get_peft_model(
    #传入模型
    model,
    #低秩矩阵的秩，也就是 A 和 B 的内部维度
    r=32,
    #指定在哪些线性层上插入 LoRA 适配器：自注意力机制中的查询、键、值、输出投影层，馈网络（MLP）中的门控、上投影、下投影层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #缩放因子，用于控制 LoRA 更新的幅度
    #通常 lora_alpha 设为与 r 相同的值，保持更新大小适中。如果太大，LoRA 的影响可能盖过原始模型；太小则效果不明显。
    lora_alpha=32,
    #不启用 dropout
    lora_dropout=0,
    #模型中的偏置项（bias）不参与训练
    bias="none",
    #梯度检查点，"unsloth"表示使用 Unsloth 库自带的、经过优化的梯度检查点实现，让重新计算的效率更高，额外时间更少
    use_gradient_checkpointing="unsloth",
    #固定随机种子，确保每次运行结果可重复
    random_state=3407,
)

# ====================== 加载数据 ======================
print("加载训练数据...")
train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

# ====================== 自定义格式化函数 ======================
def formatting_func(example):
    intent = example["user_intent"]
    history = example.get("history_up_to_prev", "（无历史对话）")
    user_input = example["current_user_input"]
    response = example["current_assistant_output"]

    prompt = f"""你是一位专业、经验丰富、温暖且非评判性的心理咨询师。你擅长根据用户的当前意图，提供针对性的支持和引导。

    当前意图：{intent}
    历史对话：{history}
    用户当前输入：{user_input}

    回复原则（严格遵守）：
    - 如果意图是“危机求助”：立即表达深刻共情和理解，稳定用户情绪，强调“你并不孤单”，强烈建议寻求专业即时帮助（如拨打心理热线），避免任何鼓励负面行为的内容，语气温柔、坚定、陪伴感强。
    - 如果意图是“情感表达”：先深度共情和接纳用户感受，使用温暖的倾听语言（如“我能感受到你的痛苦/疲惫/无助”），然后用开放式提问引导用户进一步表达和澄清情绪，帮助用户自我觉察。
    - 如果意图是“寻求建议”：在共情的基础上，给出具体、可操作、循序渐进的专业建议或应对策略，加入鼓励和赋能语言（如“你已经迈出了勇敢的一步”），避免空洞说教。

    请以专业、温暖、支持性的语气回复，只输出咨询师的回复内容，不要添加其他说明。你的回复要体现心理学专业性：使用共情、反映、澄清、开放式提问等核心技术。
    """

    return {"text": prompt + "\n\n" + response}

# 应用格式化
train_dataset = train_dataset.map(formatting_func)

# 验证集（可选）
val_dataset = load_dataset("json", data_files=VAL_FILE, split="train")
val_dataset = val_dataset.map(formatting_func)

# ====================== 训练参数 ======================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,  # 所有 checkpoint 保存这里
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    ),
)

# ====================== 开始微调 ======================
print("开始微调...")
trainer.train()

# 保存最终 LoRA 适配器
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n微调完成！")
print(f"模型保存到: {OUTPUT_DIR}")
print(f"文件夹大小约几百 MB，可直接复制到本地其他地方使用。")
print("下次加载时用：FastLanguageModel.from_pretrained('unsloth/mistral-7b-instruct-v0.3-bnb-4bit') + load_adapter('{OUTPUT_DIR}')")
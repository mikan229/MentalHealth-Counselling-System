from unsloth import FastLanguageModel
import torch

# 步骤1：加载本地原始基础模型（4bit 量化版）
# 路径改成你实际的文件夹位置
base_model_path = r"D:\PythonProject5_psy_LLM\mistral-7b-instruct-v0.3-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    base_model_path,                    # ←←← 这里用本地路径
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,                  # 保持 4bit 节省显存
)

# 步骤2：加载你微调后的 LoRA 适配器（关键！）
lora_path = r"D:\PythonProject5_psy_LLM\agent2_finetuned_model"  # ←←← 改成你的 LoRA 文件夹路径
model.load_adapter(lora_path)

# 步骤3：启用推理模式（推荐，加速生成）
model = FastLanguageModel.for_inference(model)

print("微调后模型加载成功！")

# 测试生成（用不同意图的 prompt）
prompts = [
    """当前意图：危机求助
历史对话：用户: 我最近压力很大。
用户当前输入：我真的不想活了。

请作为心理咨询师，给出专业、温暖、合适的回复。""",

    """当前意图：情感表达
历史对话：用户: 工作太累了。
用户当前输入：我每天都觉得好空虚，好累。""",

    """当前意图：寻求建议
历史对话：用户: 我失恋了。
用户当前输入：我该怎么走出这段感情？"""
]

for prompt in prompts:
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*80)
    print("Prompt:", prompt)
    print("生成的回复:")
    print(generated)
    print("="*80)

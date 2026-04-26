# 🧠 基于RAG与PEFT的双智能体心理咨询系统

> 一个融合检索增强生成（RAG）与参数高效微调（PEFT）的双智能体协作框架，面向专业、共情的心理健康对话场景。

[![License](https://img.shields.io/badge/License-MIT-green)](https://claude.ai/chat/LICENSE) [![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://python.org/) [![Model](https://img.shields.io/badge/%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B-Mistral--7B-purple)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

------

## 📖 项目简介

大语言模型在心理健康支持领域展现出巨大潜力，但直接应用于临床咨询仍面临三大挑战：

- **知识幻觉** —— 模型生成临床上不准确的内容
- **共情不足** —— 纯RAG系统回复过于刻板、缺乏温度
- **安全风险** —— 单智能体架构缺乏危机识别与干预机制

本项目通过**双智能体架构**应对上述挑战，模拟真实临床咨询流程：先评估用户意图，再匹配相应的治疗策略。

系统在五个咨询维度上经GPT-4o mini评测，相较基础模型（Mistral-7B-Instruct-v0.3）实现了 **18.26% 的性能提升**。

## 🏗️ 系统架构

```
用户输入
    │
    ▼
┌─────────────────────────────┐
│      意图分析智能体 (IAA)    │  ← DeepSeek V3.2 + Few-shot 提示
│   • 情感表达                 │
│   • 寻求建议                 │
│   • 危机求助                 │
└────────────┬────────────────┘
             │ 意图标签
             ▼
┌─────────────────────────────┐
│    回复生成智能体 (RGA)      │  ← 微调后的 Mistral-7B + RAG
│   • 上下文感知策略切换        │
│   • RAG 知识检索             │
│   • 危机安全干预             │
└────────────┬────────────────┘
             │
             ▼
        专业且富有共情的咨询回复
```

### 两个核心智能体

**意图分析智能体（IAA）**

- 主干模型：DeepSeek V3.2
- 基于 few-shot 提示对用户意图进行三分类
- 对比人工标注：整体准确率 93.26%，Cohen's Kappa 0.8042
- 危机场景召回率高达 95.8%，极低漏判率，保障用户安全

**回复生成智能体（RGA）**

- 主干模型：Mistral-7B-Instruct-v0.3（经 QLoRA 微调）
- 基于 FAISS 向量检索权威心理学文献，增强专业性
- 维护最近 4 轮对话历史，保证多轮对话连贯性
- 检测到危机意图时强制输出心理援助热线，确保安全合规

## 🗂️ 知识库构建

知识来源于权威临床文献：

|      来源      |               描述               |         文件         |
| :------------: | :------------------------------: | :------------------: |
|  **DSM-5-TR**  | 美国精神病学会诊断手册（2022版） | 29个 PDF Fact Sheets |
| **ICD-11 MMS** |     世界卫生组织国际疾病分类     |  2个 TXT + 1个 XLSX  |

**处理流程：**

- 文本切片：`RecursiveCharacterTextSplitter`（chunk_size=500，overlap=100）
- 总切片数：**3,969 个**
- 嵌入模型：`nvidia/llama-embed-nemotron-8b`（8B 参数）
- 向量索引：FAISS 近似最近邻检索
- 检索 Top-K：每次查询召回最相关的 3 个片段

## 📊 数据集构建

我们构建了一套高质量的意图标注数据集用于训练和评估 IAA：

```
PsyQA（单轮） ──┐
                 ├──► 1,040 条高质量 QA ──► DeepSeek V3.2 自动标注
CPsyCounR（多轮）┘     （人工筛选）             （few-shot 提示）
                                                      │
                                                      ▼
                                              7,108 条标注样本
                                        ┌──────────────────────────┐
                                        │ • 情感表达                │
                                        │ • 寻求建议                │
                                        │ • 危机干预                │
                                        └──────────────────────────┘
                                        训练集: 5,686 | 验证集: 711 | 测试集: 711
```

随机抽取 200 条数据由专业人工标注作为金标准，用于评估 IAA 标注质量。

## ⚙️ 微调参数配置

|       参数        |                     值                      |
| :---------------: | :-----------------------------------------: |
|     基础模型      | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` |
|     微调方法      |       QLoRA（4-bit NF4 量化 + LoRA）        |
| LoRA rank / alpha |                   32 / 32                   |
|      Dropout      |                      0                      |
|  可训练参数占比   |                    1.14%                    |
|   最大序列长度    |                4,096 tokens                 |
|      优化器       |                 8-bit AdamW                 |
|    初始学习率     |             2×10⁻⁴（线性衰减）              |
|     训练轮数      |                  3 epochs                   |
|     训练损失      |               0.1058 → 0.0153               |
|     验证损失      |               0.1187 → 0.0359               |

## 📈 实验结果

### 消融实验（GPT-4o mini 评测）

|                模型                |      平均得分       |
| :--------------------------------: | :-----------------: |
|  Mistral-7B-Instruct-v0.3（基线）  |        5.64         |
| Mistral-7B-Instruct-v0.3（仅 RAG） |        5.24         |
| Mistral-7B-Instruct-v0.3（仅微调） |        6.40         |
|      **双智能体框架（本文）**      | **6.67（+18.26%）** |

### 跨模型泛化性验证

|             模型             | 平均得分 |
| :--------------------------: | :------: |
| Qwen2.5-7B-Instruct（微调）  |   6.25   |
| Qwen2.5-7B-Instruct 双智能体 |   6.34   |
| Mistral-7B 双智能体（本文）  | **6.67** |

### IAA 意图分类性能

| 意图类别 | 精确率(%) | 召回率(%) |       F1(%)       |
| :------: | :-------: | :-------: | :---------------: |
| 情感表达 |   87.8    |   85.4    |       86.6        |
| 寻求建议 |   91.5    |   81.3    |       86.1        |
| 危机干预 |   92.7    | **95.8**  |       94.2        |
| **整体** |     —     |     —     | **准确率 93.26%** |

### 评测维度说明

模型在 200 个测试用例（单轮+多轮）上从五个维度进行评分：

1. **基础共情能力** —— 情绪识别与情感支持
2. **专业深度** —— 识别用户深层心理困扰
3. **专业能力** —— 运用 CBT、精神动力学、人本主义等咨询理论
4. **安全性** —— 危机识别与伦理边界遵守
5. **边界情况** —— 极端或模糊场景下的稳定性

## 🚀 快速上手

### 环境依赖

```bash
pip install langchain langchain-community langchain-text-splitters langchain-huggingface \
    pypdf faiss-gpu-cu12 unstructured openpyxl sentence-transformers \
    torch transformers vllm unsloth
```

**推荐硬件：** 显存 ≥ 24GB 的 NVIDIA GPU（如 RTX 4090），用于同时运行嵌入模型和微调后的生成模型。

### 项目结构

```
├── multi_agent/                          # 双智能体系统核心代码
│   ├── agent1_intent_classifier.py       # 意图分析智能体 IAA（DeepSeek V3.2）
│   ├── agent2_counselor.py               # 回复生成智能体 RGA（微调 Mistral-7B）
│   ├── agent2_rag_turned.py              # RGA + RAG 集成版本（Mistral）
│   └── agent2_rag_qwen.py                # RGA + RAG 集成版本（Qwen2.5-7B）
├── baseline_v1_allMiniLM/                # 基线 RAG 索引（all-MiniLM-L6-v2 嵌入）
├── baseline_v2_llama/
│   └── faiss_index_llama/               # 升级版 RAG 索引（llama-embed-nemotron-8b 嵌入）
├── psychology_knowledge/                 # 知识库原始文档（DSM-5-TR + ICD-11）
├── 各模型测试结果/                        # 各组实验的模型输出记录
├── memory_eval/                          # 基线模型 CMMLU 评测脚本
├── fine_tuning_mistral.py                # Mistral-7B QLoRA 微调脚本
├── fine_tuning_qwen.py                   # Qwen2.5-7B QLoRA 微调脚本
├── rag_setup.py                          # 基础 RAG 流水线（allMiniLM 版本）
├── rag_setup_llama.py                    # 升级版 RAG 流水线（llama 嵌入版本）
├── rag_full_test.py                      # RAG 系统测试脚本
├── rag_full_test_llama.py                # llama 嵌入版 RAG 测试脚本
├── test_tuned_model.py                   # 微调模型测试脚本
├── 1040data_full_labeled_ALL.jsonl       # 意图标注种子数据集（1,040 条）
└── 心理咨询大模型基准测试集.xlsx          # 五维度评测基准集（200 条）
```

> **关于 FAISS 索引：** 由于文件体积较大，索引文件无法直接上传至 GitHub。请在本地运行 `rag_setup_llama.py` 重新构建，或联系作者获取。

### 使用示例

```python
# 第一步：构建知识库索引
python rag_setup_llama.py

# 第二步：运行双智能体对话系统
from intent_agent.iaa_classifier import IntentAnalysisAgent
from response_agent.rga_generator import ResponseGenerationAgent

iaa = IntentAnalysisAgent()
rga = ResponseGenerationAgent()

dialogue_history = []
user_input = "我最近压力很大，感觉什么都做不好..."

intent = iaa.classify(user_input, dialogue_history)
response = rga.generate(user_input, intent, dialogue_history)

print(f"意图分类：{intent}")
print(f"系统回复：{response}")
```

## 🛡️ 安全机制

当 IAA 检测到**危机求助**意图（自杀倾向、自伤、暴力）时，系统将：

1. 立即覆盖正常生成流程
2. 生成充满关怀的危机回应
3. **强制输出心理援助热线信息**
4. 记录事件日志以供复核

该行为不可绕过，确保模型在边界场景下的行为符合 AI 伦理规范。

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@inproceedings{dual-agent-mental-health-2025,
  title     = {A Dual-Agent Mental Health Counseling System with RAG and PEFT},
  year      = {2025},
}
```

📌 论文审稿中，引用信息待更新。

## 🙏 致谢

- **数据集：** [PsyQA](https://aclanthology.org/2021.findings-acl.130/) · [CPsyCounR](https://aclanthology.org/2024.findings-acl.830/)
- **知识来源：** [DSM-5-TR](https://doi.org/10.1176/appi.books.9780890425787)（APA） · [ICD-11 MMS](https://icd.who.int/browse11/l-m/en)（WHO）
- **模型：** [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) · [DeepSeek V3.2](https://www.deepseek.com/) · [llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b)
- **框架：** [LangChain](https://langchain.com/) · [FAISS](https://github.com/facebookresearch/faiss) · [Unsloth](https://github.com/unslothai/unsloth)

## ⚠️ 免责声明

本系统为**学术研究原型**，不能替代专业心理健康服务。若您或身边的人正处于危机状态，请立即联系专业心理咨询师或拨打急救电话。
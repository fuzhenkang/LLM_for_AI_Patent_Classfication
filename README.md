# LLM for AI Patent Classification

该仓库提供两套大语言模型专利文本分类程序，均面向“标题、摘要、IPC 等字段判断是否为 AI 专利”的二分类或多分类任务。

```text
LLM_AIPC_v1  基于任务特定分类头的序列分类微调范式
LLM_AIPC_v2  基于 Prompt 的下一 token 预测生成式分类范式
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据格式

CSV 至少需要包含标签列和文本输入列。推荐格式：

```csv
PN,title,abstract,IPC,AI_label
CN110000001A,一种图像识别方法,本发明公开了一种基于神经网络的图像识别方法,G06N;G06V,1
CN110000002A,一种机械连接装置,本发明涉及机械零件连接结构,F16B,0
```

训练时可以使用：

```bash
--text-cols title,abstract,IPC
```

将多个字段拼接为模型输入。若已经提前拼接为 `text` 列，也可以使用：

```bash
--text-col text
```

## 分类范式

### LLM_AIPC_v1：分类头微调范式

v1 使用：

```python
AutoModelForSequenceClassification
```

流程为：

```text
文本 → LLM Backbone → 线性分类头 → 类别概率
```

该范式属于：

```text
task-specific classification head fine-tuning
sequence classification fine-tuning
判别式序列分类微调
```

适合希望把大语言模型当作特征编码器，并在顶部接入分类头的实验。

### LLM_AIPC_v2：Prompt 下一 token 分类范式

v2 使用：

```python
AutoModelForCausalLM
```

流程为：

```text
Prompt + 文本 → LLM → 下一个 token logits → 标签词“否/是”
```

该范式属于：

```text
prompt-based generative classification
next-token prediction based classification
基于提示学习的生成式分类
```

程序会构造提示词，并只比较标签词 token 的 logits，例如默认：

```text
否,是
```

## 支持模型

两套程序均通过 `--model-key` 提供默认配置：

```text
llama
qwen
glm
mistral
baichuan
```

也可以用 `--base-model` 指定具体 Hugging Face 模型，例如：

```text
Qwen/Qwen3-8B
Qwen/Qwen2.5-7B-Instruct
mistralai/Mistral-7B-v0.1
baichuan-inc/Baichuan2-7B-Base
THUDM/glm-4-9b-chat
```

## 支持微调方式

v1 和 v2 均支持：

```text
lora       # 传统 LoRA
qlora      # 量化后 LoRA，默认 4bit
rslora     # rank-stabilized LoRA
dora       # DoRA
head_only  # 冻结主干，只训练分类头或输出头参数
```

v1 的 `head_only` 表示冻结主干，只训练 `classifier/score/classification_head` 等分类头参数。  
v2 的 `head_only` 表示冻结主干，只训练语言模型输出层中与标签词打分相关的输出头参数。

## 1. 划分数据集

```bash
python split_dataset.py \
  --input data/processed/patents_cleaned.csv \
  --output-dir data/split \
  --label-col AI_label \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

输出：

```text
data/split/train.csv
data/split/valid.csv
data/split/test.csv
```

## 2. LLM_AIPC_v1 训练与评估

### 2.1 训练分类头范式 QLoRA

```bash
python LLM_AIPC_v1/llm_classifier.py \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/v1/qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --tuning-mode qlora \
  --load-in-4bit \
  --bnb-4bit-quant-type nf4 \
  --bnb-4bit-compute-dtype float16 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --batch-size 2 \
  --max-len 256 \
  --epochs 3 \
  --lr 0.00002
```

### 2.2 10 折交叉验证

```bash
python LLM_AIPC_v1/llm_classifier.py \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --data-csv data/split/train.csv \
  --output-dir outputs/v1/qwen3_8b_cv \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --cv-folds 10 \
  --tuning-mode qlora \
  --load-in-4bit \
  --batch-size 2 \
  --max-len 256 \
  --epochs 3 \
  --lr 0.00002
```

### 2.3 Optuna 参数寻优

```bash
python LLM_AIPC_v1/optuna_search.py \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/v1/optuna_qwen3_8b \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --tuning-mode qlora \
  --load-in-4bit \
  --n-trials 10 \
  --epochs 3
```

### 2.4 测试集评估

```bash
python LLM_AIPC_v1/evaluate_model.py \
  --model-dir outputs/v1/qwen3_8b_qlora \
  --test-csv data/split/test.csv \
  --output-dir outputs/evaluation/v1_qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label
```

输出：

```text
outputs/evaluation/v1_qwen3_8b_qlora/test_metrics.json
outputs/evaluation/v1_qwen3_8b_qlora/predictions.csv
```

## 3. LLM_AIPC_v2 训练与评估

### 3.1 训练 Prompt 下一 token QLoRA

```bash
python LLM_AIPC_v2/llm_classifier.py \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/v2/qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --tuning-mode qlora \
  --load-in-4bit \
  --bnb-4bit-quant-type nf4 \
  --bnb-4bit-compute-dtype float16 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --batch-size 2 \
  --max-len 256 \
  --epochs 3 \
  --lr 0.00002 \
  --save-checkpoint-steps 500
```

中断后继续训练：

```bash
python LLM_AIPC_v2/llm_classifier.py \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/v2/qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --tuning-mode qlora \
  --load-in-4bit \
  --batch-size 2 \
  --max-len 256 \
  --epochs 3 \
  --lr 0.00002 \
  --save-checkpoint-steps 500 \
  --resume-from-checkpoint outputs/v2/qwen3_8b_qlora/checkpoint-last
```

### 3.2 Optuna 参数寻优

```bash
python optuna_search.py \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/v2/optuna_qwen3_8b \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --tuning-mode qlora \
  --load-in-4bit \
  --n-trials 10 \
  --epochs 3
```

### 3.3 测试集评估

```bash
python evaluate_model.py \
  --model-dir outputs/v2/qwen3_8b_qlora \
  --test-csv data/split/test.csv \
  --output-dir outputs/evaluation/v2_qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label
```

输出：

```text
outputs/evaluation/v2_qwen3_8b_qlora/test_metrics.json
outputs/evaluation/v2_qwen3_8b_qlora/predictions.csv
```

## 4. Baichuan 与 Ministral 说明

Baichuan 的自定义模型代码可能不兼容 `BitsAndBytesConfig` 对象，v2 已对 `--model-key baichuan` 默认启用 legacy bitsandbytes 参数，并默认使用 `--device-map cuda`。

Ministral 3 / Mistral 3 不是普通 `AutoModelForCausalLM` 架构。v2 在 `--base-model` 包含 `Ministral-3` 或 `Mistral-3` 时会自动使用 `Mistral3ForConditionalGeneration`。

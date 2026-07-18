# LLM for AI Patent Classification

本仓库提供四套基于大语言模型的 AI 专利文本分类程序，均支持使用专利标题、摘要、IPC 等字段判断专利是否属于 AI 专利。

```text
LLM_AIPC_v1  基于任务特定分类头的判别式序列分类范式
LLM_AIPC_v2  基于 Prompt 的下一 token 标签词分类范式
LLM_AIPC_v3  标准 autoregressive likelihood 候选标签似然分类范式
LLM_AIPC4    与 ar_pseudo/train_gpt.py 一致的 LM-loss 训练 + 标签词 next-token 推理范式
```

四套模型共用数据集划分、Optuna 参数寻优和测试集评估脚本：

```text
split_dataset.py   划分训练集、验证集、测试集
optuna_search.py   对 v1、v2、v3 或 v4 进行验证集寻优
evaluate_model.py  自动识别 v1/v2/v3/v4 训练结果并在测试集评估
```

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 数据格式

CSV 至少需要包含标签列和文本输入列。推荐格式如下：

```csv
PN,title,abstract,IPC,AI_label
CN110000001A,一种图像识别方法,本发明公开了一种基于神经网络的图像识别方法,G06N;G06V,1
CN110000002A,一种机械连接装置,本发明涉及机械零件连接结构,F16B,0
```

多字段输入可以写成：

```bash
--text-cols title,abstract,IPC
```

如果已经提前拼接为 `text` 列，也可以写成：

```bash
--text-col text
```

标签列可以是 `0/1`，也可以是文字标签。程序会自动编码标签。对于 v2/v3/v4，`--label-words` 的顺序需要与自动编码后的类别顺序一致；为了避免中文标签词被拆成多个 token，推荐优先使用 `No,Yes`。

## 3. 四种分类范式

### LLM_AIPC_v1：分类头微调

v1 使用 `AutoModelForSequenceClassification`：

```text
文本 -> LLM Backbone -> 线性分类头 -> 类别 logits -> 分类交叉熵损失
```

### LLM_AIPC_v2：下一 token 标签词分类

v2 使用 `AutoModelForCausalLM`，训练和推理都只关注最后位置的标签词 logits：

```text
Prompt + 文本 -> LLM -> 下一 token logits -> 取 No/Yes logits -> 分类交叉熵损失
```

### LLM_AIPC_v3：标准 AR 候选标签似然分类

v3 使用 `AutoModelForCausalLM`，枚举每个候选标签并计算完整候选序列的自回归语言模型似然：

```text
候选序列 1: Label:No  + Text:专利文本 -> LM loss -> score(No)
候选序列 2: Label:Yes + Text:专利文本 -> LM loss -> score(Yes)
预测类别 = score 最高的候选标签
```

默认使用平均 token 负损失作为候选分数：

```bash
--likelihood-reduction mean
```

### LLM_AIPC4：ar_pseudo 风格

v4 的逻辑与 `amazon-science/Generative-vs-Discriminative-Classifiers` 中的 `ar_pseudo/train_gpt.py` 保持一致：

```text
训练：Text:专利文本 Label:真实标签词 -> Causal LM 全序列 next-token 交叉熵损失
推理：Text:专利文本 Label: -> 取最后位置 No/Yes 标签词 logits -> 分类
```

因此 v4 的训练损失不是 v2 的“标签词 logits 分类交叉熵”，而是 Hugging Face Causal LM 返回的 `outputs.loss`，即自回归语言模型的 token-level cross entropy loss。

## 4. 支持模型与微调方式

通过 `--model-key` 使用默认配置：

```text
llama
qwen
deepseek
glm
mistral
baichuan
```

也可以通过 `--base-model` 指定具体 Hugging Face 模型，例如：

```text
Qwen/Qwen3-8B
Qwen/Qwen2.5-7B-Instruct
deepseek-ai/deepseek-llm-7b-base
mistralai/Mistral-7B-v0.1
baichuan-inc/Baichuan2-7B-Base
THUDM/glm-4-9b-chat
```

v1、v2、v3 和 v4 均支持：

```text
lora       传统 LoRA
qlora      量化后 LoRA，默认 4bit
rslora     rank-stabilized LoRA
dora       DoRA
head_only  冻结主干，只训练分类头或输出头参数
```

## 5. 划分数据集

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

## 6. LLM_AIPC_v1 训练

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
  --batch-size 2 \
  --gradient-steps 4 \
  --max-len 256 \
  --epochs 3 \
  --lr 0.00002 \
  --save-checkpoint-steps 500
```

## 7. LLM_AIPC_v2 训练

```bash
python LLM_AIPC_v2/llm_classifier.py \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/v2/qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --label-words No,Yes \
  --tuning-mode qlora \
  --load-in-4bit \
  --batch-size 2 \
  --gradient-steps 4 \
  --max-len 256 \
  --epochs 3 \
  --lr 0.00002 \
  --save-checkpoint-steps 500
```

## 8. LLM_AIPC_v3 训练

```bash
python LLM_AIPC_v3/llm_classifier.py \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/v3/qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --label-words No,Yes \
  --likelihood-reduction mean \
  --tuning-mode qlora \
  --load-in-4bit \
  --batch-size 1 \
  --gradient-steps 8 \
  --max-len 256 \
  --epochs 3 \
  --lr 0.00002 \
  --save-checkpoint-steps 500
```

## 9. LLM_AIPC4 训练

```bash
python LLM_AIPC4/llm_classifier.py \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/v4/qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --label-words No,Yes \
  --tuning-mode qlora \
  --load-in-4bit \
  --bnb-4bit-quant-type nf4 \
  --bnb-4bit-compute-dtype float16 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --batch-size 2 \
  --gradient-steps 4 \
  --max-len 256 \
  --epochs 3 \
  --lr 0.00002 \
  --save-checkpoint-steps 500
```

断点续训时继续指定同一个输出目录，并增加：

```bash
--resume-from-checkpoint outputs/v4/qwen3_8b_qlora/checkpoint-last
```

## 10. Optuna 参数寻优

v4 寻优示例：

```bash
python optuna_search.py \
  --classifier-version v4 \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/optuna/v4_qwen3_8b \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --label-words No,Yes \
  --tuning-mode qlora \
  --load-in-4bit \
  --n-trials 10 \
  --epochs 3
```

输出中包含：

```text
best_params.json
optuna_trials.json
best_model/
```

`best_model/` 是验证集指标最优 trial 的模型目录，可直接用于测试集评估。

## 11. 测试集评估

v1、v2、v3 和 v4 共用同一个评估脚本，程序会根据 `model-dir/config.json` 自动识别模型范式。

```bash
python evaluate_model.py \
  --model-dir outputs/v4/qwen3_8b_qlora \
  --test-csv data/split/test.csv \
  --output-dir outputs/evaluation/v4_qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label
```

评估 Optuna 找到的最优模型：

```bash
python evaluate_model.py \
  --model-dir outputs/optuna/v4_qwen3_8b/best_model \
  --test-csv data/split/test.csv \
  --output-dir outputs/evaluation/v4_qwen3_8b_best \
  --text-cols title,abstract,IPC \
  --label-col AI_label
```

输出：

```text
test_metrics.json
predictions.csv
```

## 12. 梯度累积

`--gradient-steps` 表示梯度累积步数。有效 batch size 的计算方式为：

```text
有效 batch size = batch-size * gradient-steps
```

例如 `--batch-size 2 --gradient-steps 4` 等效于每 8 条样本更新一次参数，但单次显存占用仍接近 `batch-size 2`。

## 13. Baichuan 与 Ministral 说明

Baichuan 的自定义模型代码可能不兼容 `BitsAndBytesConfig` 对象。程序对 `--model-key baichuan` 默认启用 legacy bitsandbytes 参数，并默认使用 `--device-map cuda`，以减少 CPU/GPU 张量不在同一设备的问题。

Ministral 3 / Mistral 3 不是普通 `AutoModelForCausalLM` 架构。程序在 `--base-model` 包含 `Ministral-3` 或 `Mistral-3` 时会自动使用 `Mistral3ForConditionalGeneration`。

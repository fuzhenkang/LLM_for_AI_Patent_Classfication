# LLM for AI Patent Classification

本仓库提供两套基于大语言模型的 AI 专利文本分类程序，均支持使用专利标题、摘要、IPC 等字段判断专利是否属于 AI 专利。

```text
LLM_AIPC_v1  基于任务特定分类头的序列分类微调范式
LLM_AIPC_v2  基于 Prompt 的下一 token 预测生成式分类范式
```

两套模型共用数据集划分、Optuna 参数寻优和测试集评估脚本：

```text
split_dataset.py   划分训练集、验证集、测试集
optuna_search.py   对 v1 或 v2 进行验证集寻优
evaluate_model.py  自动识别 v1/v2 训练结果并在测试集评估
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

标签列可以是 `0/1`，也可以是文字标签。程序会自动编码标签。

## 3. 两种分类范式

### LLM_AIPC_v1：分类头微调

v1 使用 `AutoModelForSequenceClassification`：

```text
文本 -> LLM Backbone -> 线性分类头 -> 类别 logits -> 交叉熵损失
```

这是常见的 sequence classification fine-tuning，也就是在大语言模型顶部接任务特定分类头进行判别式分类。

### LLM_AIPC_v2：下一 token 分类

v2 使用 `AutoModelForCausalLM`：

```text
Prompt + 文本 -> LLM -> 下一 token logits -> 标签词得分 -> 交叉熵损失
```

这是 prompt-based generative classification。程序会构造提示词，并只比较标签词对应 token 的 logits，例如默认标签词为：

```text
否,是
```

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

v1 和 v2 均支持：

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

输出：

```text
data/split/train.csv
data/split/valid.csv
data/split/test.csv
```

训练集用于模型微调，验证集用于选择参数和最佳模型，测试集只用于最终评估。

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

断点续训：

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
  --save-checkpoint-steps 500 \
  --resume-from-checkpoint outputs/v1/qwen3_8b_qlora/checkpoint-last
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
  --label-words 否,是 \
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

断点续训：

```bash
python LLM_AIPC_v2/llm_classifier.py \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/v2/qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --label-words 否,是 \
  --tuning-mode qlora \
  --load-in-4bit \
  --batch-size 2 \
  --gradient-steps 4 \
  --max-len 256 \
  --epochs 3 \
  --lr 0.00002 \
  --save-checkpoint-steps 500 \
  --resume-from-checkpoint outputs/v2/qwen3_8b_qlora/checkpoint-last
```

## 8. Optuna 参数寻优

v1 寻优：

```bash
python optuna_search.py \
  --classifier-version v1 \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/optuna/v1_qwen3_8b \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --tuning-mode qlora \
  --load-in-4bit \
  --n-trials 10 \
  --epochs 3
```

v2 寻优：

```bash
python optuna_search.py \
  --classifier-version v2 \
  --model-key qwen \
  --base-model Qwen/Qwen3-8B \
  --train-csv data/split/train.csv \
  --valid-csv data/split/valid.csv \
  --output-dir outputs/optuna/v2_qwen3_8b \
  --text-cols title,abstract,IPC \
  --label-col AI_label \
  --label-words 否,是 \
  --tuning-mode qlora \
  --load-in-4bit \
  --n-trials 10 \
  --epochs 3
```

输出中会包含：

```text
best_params.json
optuna_trials.json
best_model/
```

`best_model/` 是验证集指标最优 trial 的模型目录，可直接用于测试集评估。

`--gradient-steps` 表示梯度累积步数。有效 batch size 的计算方式为：

```text
有效 batch size = batch-size × gradient-steps
```

例如 `--batch-size 2 --gradient-steps 4` 等效于每 8 条样本更新一次参数，但单次显存占用仍接近 `batch-size 2`。

## 9. 测试集评估

v1 和 v2 共用同一个评估脚本，程序会根据 `model-dir/config.json` 自动识别模型范式。

```bash
python evaluate_model.py \
  --model-dir outputs/v1/qwen3_8b_qlora \
  --test-csv data/split/test.csv \
  --output-dir outputs/evaluation/v1_qwen3_8b_qlora \
  --text-cols title,abstract,IPC \
  --label-col AI_label
```

如果评估 Optuna 找到的最优模型：

```bash
python evaluate_model.py \
  --model-dir outputs/optuna/v2_qwen3_8b/best_model \
  --test-csv data/split/test.csv \
  --output-dir outputs/evaluation/v2_qwen3_8b_best \
  --text-cols title,abstract,IPC \
  --label-col AI_label
```

输出：

```text
test_metrics.json
predictions.csv
```

## 10. Baichuan 与 Ministral 说明

Baichuan 的自定义模型代码可能不兼容 `BitsAndBytesConfig` 对象。v2 对 `--model-key baichuan` 默认启用 legacy bitsandbytes 参数，并默认使用 `--device-map cuda`，以减少 CPU/GPU 张量不在同一设备的问题。

Ministral 3 / Mistral 3 不是普通 `AutoModelForCausalLM` 架构。v2 在 `--base-model` 包含 `Ministral-3` 或 `Mistral-3` 时会自动使用 `Mistral3ForConditionalGeneration`。

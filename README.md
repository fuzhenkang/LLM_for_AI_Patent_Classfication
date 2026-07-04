# LLM next-token 大语言模型分类

该项目提供独立的大语言模型指令模板 + 预测下一个 token 文本分类流程，不使用交叉验证。

安装依赖：

```powershell
pip install -r requirements.txt
```

流程为：

1. 将数据集划分为训练集、验证集和测试集。
2. 使用训练集训练模型。
3. 根据验证集效果选择最佳 checkpoint 或进行 Optuna 参数寻优。
4. 最后只在测试集上评估最终模型。

模型使用 `AutoModelForCausalLM`，输入指令模板后取最后一个位置的 logits，并只比较标签词 token，例如默认 `否,是`。损失函数为分类常用的 `CrossEntropyLoss`。

## 数据格式

CSV 至少需要包含标签列和模型输入列。推荐字段如下：

```csv
PN,title,abstract,IPC,label
CN110000001A,一种图像识别方法,本发明公开了一种基于神经网络的图像识别方法,G06N;G06V,1
CN110000002A,一种机械连接装置,本发明涉及机械零件连接结构,F16B,0
```

训练时可以用 `--text-cols title,abstract,IPC` 将标题、摘要和 IPC 拼接为模型输入。若已经提前清洗并拼接好了 `text` 字段，也可以继续使用 `--text-col text`。

## 支持模型

通过 `--model-key` 支持：

```text
llama
qwen
glm
mistral
baichuan
```

也可以用 `--base-model` 指定具体 Hugging Face 模型，例如：

```text
Qwen/Qwen2.5-7B-Instruct
mistralai/Mistral-7B-v0.1
baichuan-inc/Baichuan2-7B-Base
THUDM/glm-4-9b-chat
```

## 支持微调方式

通过 `--tuning-mode` 选择：

```text
lora       # 传统 LoRA
qlora      # 量化后 LoRA，默认 4bit
rslora     # rank-stabilized LoRA
dora       # DoRA
head_only  # 冻结主干，只训练输出头参数
```

在 next-token 分类范式中，`head_only` 表示只调整语言模型输出层对标签词的打分。

## 1. 划分数据集

```powershell
python split_dataset.py `
  --input data\processed\patents_cleaned.csv `
  --output-dir data\split `
  --label-col label `
  --train-ratio 0.8 `
  --valid-ratio 0.1 `
  --test-ratio 0.1 `
  --seed 42
```

## 2. 训练 QLoRA 大语言模型分类器

```powershell
python llm_classifier.py `
  --model-key qwen `
  --base-model Qwen/Qwen2.5-7B-Instruct `
  --train-csv data\split\train.csv `
  --valid-csv data\split\valid.csv `
  --output-dir outputs\llm\qwen_qlora `
  --text-cols title,abstract,IPC `
  --label-col label `
  --tuning-mode qlora `
  --load-in-4bit `
  --bnb-4bit-quant-type nf4 `
  --bnb-4bit-compute-dtype float16 `
  --lora-r 16 `
  --lora-alpha 32 `
  --lora-dropout 0.05 `
  --batch-size 2 `
  --max-len 256 `
  --epochs 3 `
  --lr 0.00002
```

## 3. 验证集 Optuna 寻优

```powershell
python optuna_search.py `
  --model-key qwen `
  --base-model Qwen/Qwen2.5-7B-Instruct `
  --train-csv data\split\train.csv `
  --valid-csv data\split\valid.csv `
  --output-dir outputs\optuna\qwen_qlora `
  --text-cols title,abstract,IPC `
  --label-col label `
  --tuning-mode qlora `
  --load-in-4bit `
  --n-trials 10 `
  --epochs 3
```

输出中 `best_params.json` 的 `best_model_dir` 是验证集表现最好的 trial 模型目录。

## 4. 测试集评估

```powershell
python evaluate_model.py `
  --model-dir outputs\llm\qwen_qlora `
  --test-csv data\split\test.csv `
  --output-dir outputs\evaluation\qwen_qlora `
  --text-cols title,abstract,IPC `
  --label-col label
```

输出：

```text
outputs/evaluation/qwen_qlora/test_metrics.json
outputs/evaluation/qwen_qlora/predictions.csv
```

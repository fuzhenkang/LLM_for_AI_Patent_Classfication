# KE-WSPI 完整统一工程版（科研实现）

**Knowledge-Enhanced Weakly Supervised Patent Identification**

本目录一次性整合 **本地知识检索 + 实际大语言模型 API 自动调用 + 证据约束弱标注 + 具有成对依赖势函数的半监督概率标签模型 + 主动学习反馈循环 + SciBERT/PatentBERT 兼容的神经软标签蒸馏 + 双向 Spy 复现实验与消融对比**。

> **重要的科研真实性说明：** 这是完整架构的代码实现，不是已经在真实专利和GPU环境下验证有效的学术成果。LLM服务需用户的模型API密钥与调用额度，SciBERT/PatentBERT需下载模型权重和GPU/CPU资源，真实专家标签及专利数据需要用户提供。未公开的原文人工CPC审查决定无法仅靠编写程序“严格”还原，故Spy实现标记为 **公开方法重建** 而非原始结果完全复现。

## 项目结构

| 文件 | 内容 |
|---|---|
| `rag_llm.py` | 本地TF-IDF词项+字符混合检索、OpenAI兼容Chat Completions接口、重试、缓存及原文引文核验 |
| `label_graph.py` | 学习源准确率/类别关系的稀疏成对依赖能量模型，专家锚定的半监督MLE |
| `neural_distill.py` | 可加载HuggingFace序列分类编码器的教师概率KL蒸馏、温度与专家硬标签联合损失 |
| `spy_baseline.py` | 双向Spy注入、85%/99%分位数过滤，记录人工CPC筛选缺失 |
| `run.py` | 单入口，实验对比、主动选择与在线LLM核验、融合重估、神经模型训练和输出 |
| `example_config.json` | 全部配置，无密钥明文 |
| `examples/` | 可运行的合成样本与本地技术知识 |
| `tests/` | 源依赖模型、RAG规则与检索的测试 |

## 1. 环境

Python 3.10+，建议 Python 3.11。对于CUDA，请按照本机显卡与PyTorch官方说明安装匹配的PyTorch。

```powershell
cd KE-WSPI-full
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Windows如遇到 `python` 命令未找到，可尝试 `py -3.11`。

## 2. 立即执行离线端到端示例

```powershell
python examples/generate_demo.py
python run.py --config example_config.json
```

结果写入 `outputs/demo/`：
- `experiments.csv`：每轮多数投票、固定条件独立融合、稀疏依赖融合及可用时的Spy对照指标；
- `round_X_teacher.csv`：各轮用于学生训练的融合概率标签；
- `review_round_X.csv`：待专家复核的主动学习队列；
- `split.json`：测试集与训练集ID；
- `cache/`：已核验LLM请求的结果缓存。

合成示例的“专家Oracle”用真实标签模拟复核；此方法**只允许用于演示或离线标注数据集**，真实未标注语料必须将 `auto_oracle` 设为 `false`。

## 3. 启用真正的LLM与RAG自动调用

修改 `example_config.json`：

```json
"llm": {
  "enabled": true,
  "base_url": "https://api.openai.com/v1",
  "model": "请填写实际可用模型名",
  "api_key_env": "OPENAI_API_KEY",
  "timeout": 90,
  "max_calls_per_round": 12
}
```

PowerShell设置密钥：

```powershell
$env:OPENAI_API_KEY="填写你的密钥"
python run.py --config example_config.json
```

或将 `base_url` 改成你的兼容API服务地址、`model`改成实际支持的模型。**代码会实际发送用于核验的专利文本与检索知识到指定服务**，请确保有权发送相关数据。无需自行写LLM调用脚本，成功返回的结果将被验证并缓存。密钥仅从环境变量读取，绝不可提交GitHub。

检索使用 `knowledge.json` 作为本地技术知识库，可替换为更完整的WIPO/CPC与人工标注技术定义。当前算法为稀疏文本混合检索，不包含向量数据库服务、交叉编码重排，也不应将其称为经过验证的完整生产级RAG。

## 4. 启用SciBERT / PatentBERT神经蒸馏

学生模型配置：

```json
"student": {
  "model_name": "allenai/scibert_scivocab_uncased",
  "epochs": 2,
  "batch_size": 8,
  "learning_rate": 0.00002,
  "max_length": 256,
  "temperature": 2.0,
  "distill_weight": 0.7
}
```

```bash
python run.py --config example_config.json --neural
```

`--neural` 会在主动学习轮次后，使用 **ECLR融合后验概率作为教师软标签** 训练Hugging Face二分类神经学生，配合专家锚点联合KL与硬标签交叉熵。改变 `model_name` 可测试兼容的PatentBERT模型ID或已下载的本地权重路径。并非所有名为PatentBERT的仓库都公开了可直接加载的序列分类权重；具体模型必须事先验证可下载且许可允许使用。

模型缓存和显卡依赖属于实际运行环境要求；离线环境会因无法下载权重而报错，代码不会虚构模型训练成功。

## 5. 依赖标签概率模型的算法定义

实现 `label_graph.py` 的可学习参数：
- 一元势函数 `unary[g,y,j,l]`：技术组别、潜类别、来源和来源输出；
- 成对势函数 `pair[(j,k),y,l_j,l_k]`：关联标签函数共同依赖；
- 类别先验 `prior_logits`；
- 用 `3^m` 枚举完整弱标签配置计算类别条件配分函数，对潜在真实标签边缘化；
- 通过专家已标注子集的似然与未标注数据的边缘似然共同训练；
- `pair_l2` 对成对势函数进行惩罚。

此方式是真实实现的依赖参数训练，**但不保证来源参数在任何弱监督分布下都可识别**。当前精确配分函数针对不多于8个弱标签来源，增加更多来源应改成近似推断或分组标签函数。默认pair列表用于CPC与关键词的相关性，需要基于实际数据重新设计。

## 6. 主动学习与闭环策略

每轮执行：
1. 生成弱标签及已有LLM证据标签；
2. 使用训练集内的专家锚点重估依赖概率标签模型；
3. 对所有训练样本推断融合概率，并训练轻量分类器；
4. 对未核验样本用 **概率不确定性+弱标签分歧** 选取高价值复核队列；
5. `auto_oracle=true` 时，在具备完整真值的离线数据中模拟审核，加入专家锚点；
6. `auto_oracle=false` 时只导出审核队列并结束当前运行，不伪造专家判断；下次需将审核结果合并到数据标签后重新运行；
7. 启用 `--neural` 时启动神经蒸馏。

启用在线LLM时自动处理有限预算的待核验样本，LLM结果作为弱标签参与下一轮融合；LLM的预测不自动冒充专家金标准。

## 7. Spy复现边界

`spy_baseline.py` 重建两轮Spy机制，分别按照85%推定负Spy与99%推定正Spy识别率设置分位数阈值。

**仍非原文严格逐项复现**：原文的人工CPC排除、训练数据版本、专利族去重结果、专家判断和部分学习器配置未必均可公开获得。该模块给出清晰可审计的过程记录；科研正式对照应获取作者公开源码及筛选名单，与原算法逐环节比对。不能将两种实现的性能差异全部归因于KE-WSPI。

## 8. 论文级消融和统计检验

内置统一测试划分下的 `majority`、`independent`、`dependent` 以及可用的 `spy_reconstruction`。神经蒸馏可通过 `--neural` 开关比较，在线LLM可通过 `llm.enabled` 对比。

**尚需真实数据支持的检验：** 不同随机种子、按专利族/申请年份/专利局隔离划分、人工标注可靠性、来源依赖干预、相同标注预算与成本、真实世界准确率校准以及对原文作者代码的逐项严格复现。内置随机行切分仅是演示性实验协议。

## 9. 数据格式

UTF-8 CSV字段：

`patent_id,title,abstract,claims,cpc,family_id,year,label`

- `patent_id`唯一；
- `label`为专家标注0/1；未知留空；
- 至少40条正负两类专家标注样本用于开发与测试；
- 真实数据 `auto_oracle=false`；
- CPC样例只有G06N和少量排除词，仅是标签函数的可扩展演示。正式研究务必导入完整领域分类定义，并建立高质量困难负样本池。

## 10. 常见问题

**为什么不提供任何论文可直接使用的准确率结论？** 实际LLM账号、模型权重、标注集与所用专利语料因人而异，模型运行和科研显著性必须基于真实数据完成，不能由自动生成的示例替代。

**这是不是“已经完整验证成功”？** 不是。项目包括全部所要求的算法模块和可执行入口，但真实外部服务与模型下载尚需使用者的资源与配置。

**能否替代原文Spy完整复现？** 不能保证，尤其是原始人工过滤规则无法从论文文字自动恢复。

## 参考文献

Wu, T., Min, C., Ding, W. W., Wang, G., & Zhang, K. (2026). *Data and methods for identifying artificial intelligence-related patents*. Research Policy, 55, 105599. https://doi.org/10.1016/j.respol.2026.105599


## 11. 可配置混合知识检索（RAG更新）

RAG已新增 `hybrid_retrieval.py`：BM25、SentenceTransformers Embedding + FAISS、混合RRF及可选Cross-Encoder Reranker；索引使用知识内容哈希持久化。项目配置增加顶层 `retrieval.mode`（`bm25`/`dense`/`hybrid`）及模型、候选数、缓存目录。**详细步骤与消融设计参见 [README_RAG.md](README_RAG.md)**。

注意：默认示例配置的 `hybrid` 模式首次运行需下载Embedding模型；离线环境可临时设为 `bm25`。当前检索结果仍需外部LLM API完成核验，不应将相似度直接视为标签置信度。

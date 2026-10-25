# KE-WSPI：知识增强弱监督专利识别算法

**Knowledge-Enhanced Weakly Supervised Patent Identification (KE-WSPI)**

本目录是独立的 Python **研究原型**，不修改仓库既有的 LLM_AIPC_v1–v4。目的在于验证：通过CPC/关键词弱标签、LLM证据核验、专家锚定的EM概率融合，能否改善新兴技术专利识别的训练标签质量。

> **版本与限制（请先阅读）**
>
> - 已实现：规则标签函数、原文证据引用匹配、条件化来源可靠性近似估计、EM后验软标签、来源折叠以减轻重复计票、Spy启发式对照、统一学生分类器与消融评估。
> - 未实现：完整稀疏图概率依赖势函数、RAG自动检索、真实LLM API请求、在线主动学习回写ECLR、神经网络知识蒸馏、PatentBERT/SciBERT微调、全文技术证据语义蕴含验证、原文Spy方法严格复现。这些不能在论文中声称已完成。
> - **仅为可扩展的实验脚手架**：没有真实标注数据时，运行成功不代表性能更优。演示数据不能用于科研结论。

## 1. 目录

```text
KE-WSPI/
├── kewspi.py                 # 弱标注、证据读取、ECLR概率融合、Spy近似对照和实验
├── active_learning.py         # 主动学习策略、离线学习曲线和人工复核队列
├── distillation.py            # 教师概率软标签与轻量学生蒸馏消融
├── README_V2.md               # V2详细中文运行说明
├── requirements.txt
├── examples/create_demo.py  # 人工合成烟雾测试数据
├── tests/test_core.py        # 基础测试
├── .gitignore
└── README.md
```

## 2. 算法模块

### M1：规则弱监督标签函数

输入专利标题、摘要、权利要求与CPC。三个标签函数分别检测：G06N类CPC、AI相关术语、非AI电力变压器等排除语境。每个函数输出 **-1=非AI，0=弃权，+1=AI**；没有命中AI词汇不被自动判为非AI。代码中的关键词仅用于演示，**不等于WIPO公布的完整CPC分类清单**。

### M2：LLM证据结果接入

可以读取预先准备的JSONL文件（由外部RAG/LLM程序产生）：

```json
{"patent_id":"P001","label":1,"evidence":[{"quote":"a neural network performs classification"}],"reason":"AI participates in technical implementation"}
```

只有**每段引用都能在本专利原文中精确匹配**时才接收其标签，其余弃权。注意：原文匹配不代表语义推断正确，当前没有提供独立NLI核验或专利技术作用的严格判断；建议另外制作人工审查队列。

### M3：ECLR 概率标签融合

设弱标签矩阵 `L`，专家锚定标签 `y_G`，可选CPC组别 `z` 与证据有效性 `e`，使用带Dirichlet式平滑与全局收缩的两类潜变量生成模型，迭代估计 `q_i=P(Y_i=1|L_i,z_i,e_i)`。条件独立假设仍存在；相关标签函数使用 `collapsed` 变体做保守组级折叠，**不是**统计上充分的依赖建模。

`ECLR` 类接受 `group_mode` 和 `evidence_mode` 开关，以支持必要的消融。专家锚点阻止标签模型自由交换潜在类别。

### M4：统一轻量分类

所有实验使用 TF-IDF + LogisticRegression 学生模型。软标签使用两份正/负复制样本的加权对数损失实现，并用专家锚点替换相应的软目标。**目前未实现神经网络蒸馏**。

## 3. 安装

Python 3.10+：

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## 4. 最快运行

```bash
python examples/create_demo.py
python kewspi.py --data examples/demo.csv --out results/demo --seed 42
```

合成数据极易分辨，所有算法都可能获得近乎满分结果；此现象仅证明执行流程通顺。输出：

- `results/demo/metrics.csv`：各基准指标和运行状态；
- `results/demo/predictions.csv`：独立测试子集上每件专利的预测；
- `results/demo/splits.json`：专家锚点、开发和测试ID列表。

## 5. 真实数据

输入 UTF-8 CSV 必须包含：

| 字段 | 含义 |
|---|---|
| patent_id | 唯一专利标识 |
| abstract | 专利摘要 |
| title | 标题，可为空 |
| claims | 权利要求，可为空 |
| cpc | 多个代码可用分号分隔 |
| label | 专家真实标签：1为AI、0为非AI；未知留空 |

实验要求至少30条已知专家标签，并且包含正负两类。脚本通过分层随机拆分约60%/20%/20%的专家标签作训练锚点/开发/测试；未标注样本可以参与弱监督学习。**当前按专利行随机拆分**，没有专利族、时间或地域隔离。正式论文必须自行补充专利族去重与时间外、技术外、专利局外评估，并独立控制专家标注的代表性。

运行：

```bash
python kewspi.py --data data/patents.csv --out results/real --seed 42
```

如已从外部RAG/LLM完成证据标注，可以加入：

```bash
python kewspi.py --data data/patents.csv --llm-jsonl data/llm_labels.jsonl --out results/llm
```

本仓库不会主动发送任何专利文本至外部API，也不保存密钥。

## 6. 对比与消融方法

| 名称 | 对照意义 | 实际实现 |
|---|---|---|
| `majority` | 无可靠性建模 | 弱标签多数投票 |
| `fixed` | 经典弱监督基础版 | 固定来源可靠性EM |
| `group` | 技术条件化可靠性 | 基于CPC前三位的条件分组收缩 |
| `evidence` | 证据约束的增量 | 按LLM引用匹配情况调节LLM标签来源 |
| `collapsed` | 相关来源依赖消融 | CPC和关键词先折叠后融合 |
| `spy_proxy` | 原文Spy思想的对照 | 两轮Spy启发式概率筛选 |

**Spy严格性声明：** `spy_proxy` 并非Wu等（2026）的原始代码，使用不同的训练分类器、动态Spy规模、简化分位数阈值与无人工CPC复查流程。不得将其标为“原文Spy严格复现”。进行论文对照时应独立实现或获取原文的完整代码、筛选样本、分类阈值及人工过滤口径。

**实验公平性声明：** 所有变体使用同一测试ID和同一最终分类器，但当前没有严格匹配训练样本覆盖率、复核预算、LLM调用成本和真实类别总体比例。输出为初步可运行对照，尚不能支持学术意义上的因果归因与性能优势结论。

## 7. 模型解释和风险

- LLM引用精确匹配仅确保片段存在，不保证技术判断正确。
- CPC/关键词与LLM的判断可能依赖同一术语；使用`collapsed`只是一种可检验的依赖近似。
- 使用少量专家锚点避免无监督标签模型不可识别的问题；开发集和测试集不可被规则优化或词典构建过程读取。
- 真实数据通常类别极不平衡；应报告Precision、Recall、F1、PR-AUC、校准指标与相同覆盖率下的标签误差。
- 当前实现的先验为演示固定值0.15，正式实验需根据代表性人工抽样估计并校准总体类先验。
- 时间外测评应锁定训练时期可用的技术知识版本，避免未来数据泄漏。

## 8. 推荐的正式科研扩展

**近期**：将完整WIPO/CPC技术分类导入标签函数，增加专家证据标注；将RAG+LLM生成器（检索、技术功能抽取、NLI验证、失败重试）独立实现；针对LLM标签源学习证据分数与准确率之间的统计关系。

**中期**：在当前EM模型中加入经过专家标注约束的稀疏成对来源依赖势函数；对技术组别进行层级参数共享；加入概率校准、标签覆盖率—错误率曲线和多随机种子评估。

**后期**：接入PatentBERT/SciBERT/CNN Text CA、主动学习复核与知识蒸馏；对接真实专利总体并采用专利族与时间隔离开展严格Spy复现和下游创新情报分析。

## 9. 参考

Wu, T., Min, C., Ding, W. W., Wang, G., & Zhang, K. (2026). *Data and methods for identifying artificial intelligence-related patents*. Research Policy 55, 105599. DOI: https://doi.org/10.1016/j.respol.2026.105599

### 关于研究结果

此仓库不附带任何真实专利精度结果。所有报告须注明实际数据源、分类定义、抽样协议、参数、专家标签及实验版本。请勿上传包含私有专利数据或API密钥的文件。


## 10. V2 新增：主动学习与软标签蒸馏

**新增的两个实验脚本**已经上传到相同分支，详情参见 [README_V2.md](README_V2.md)。主动学习提供随机、熵、不一致性和混合采样，支持离线专家真值模拟及导出复核队列；蒸馏使用外部教师概率和专家锚点训练TF-IDF+LogisticRegression轻量学生，并提供四种监督/蒸馏变体。

快速体验：

```bash
python examples/create_demo.py
python examples/create_teacher_demo.py
python active_learning.py --data examples/demo.csv --out results/al_demo
python distillation.py --data examples/demo.csv --teacher examples/teacher_demo.csv --out results/kd_demo
```

**务必区分**：已实现的是离线主动学习实验和概率软标签蒸馏基线。尚未实现真实专家/LLM在线反馈后自动重估ECLR的完整闭环，也未实现PatentBERT/SciBERT的神经知识蒸馏。示例教师概率由合成数据的真实标签构造，只允许用来验证程序运行，不得作为论文实验数据。


## 11. V3 ECLR—主动复核—学生训练闭环

新增 [closed_loop.py](closed_loop.py)，实现同轮 ECLR 标签融合、主动选样、人工反馈或离线 Oracle、专家标签锚定回写及重新训练学生模型的循环。使用流程、输出、实验边界详见 [README_V3.md](README_V3.md)。

```bash
python examples/create_demo.py
python closed_loop.py --data examples/demo.csv --mode oracle --out results/closed_demo --rounds 3 --budget 10
```

这是一套**离线可执行的完整流程原型**，并非已部署的RAG自动标注服务，也不是完整神经蒸馏实现。真实人工反馈通过复核CSV传入；测试数据被隔离；Spy严格复现和证据蕴含核验仍需扩展。

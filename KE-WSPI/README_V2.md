# KE-WSPI V2：主动学习与软标签蒸馏补充说明

本补充说明对应独立脚本 `active_learning.py` 和 `distillation.py`，依赖仓库已有 `kewspi.py` 与 `requirements.txt`。两个模块已经编码，但不能替代真实的RAG大模型生成器、专家复核系统或神经网络知识蒸馏器。

## 1. 一键合成演示

从 `KE-WSPI` 目录执行：

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python examples/create_demo.py
python examples/create_teacher_demo.py
python active_learning.py --data examples/demo.csv --out results/al_demo --initial 20 --batch 10 --rounds 5 --seed 42
python distillation.py --data examples/demo.csv --teacher examples/teacher_demo.csv --out results/kd_demo --seed 42
```

合成演示标签与教师概率由相同规则生成，**只用于测试程序可运行，不能用来证明主动学习或蒸馏有效**。

## 2. 主动学习 `active_learning.py`

提供四种样本选择：
- `random`：固定随机种子选样；
- `entropy`：二元熵最高优先；
- `disagreement`：非弃权规则正负投票分歧；
- `hybrid`：熵 0.5 + 分歧 0.3 + 语义新颖性 0.2；
并保留 `novelty` 单独采样接口。`select_batch` 只从 eligible 未复核索引中选择，严格遵守预算。新颖性由当前待选文本与已复核文本之间最大余弦相似度确定。**以上权重是原型默认值，必须在独立开发集或事先预注册确定，不能根据测试集修改。**

`offline_learning_curve` 使用完全具备专家真值的数据作离线模拟：先分出25%独立测试集，之后在余下的查询池中选择初始已标注样本，迭代训练TF-IDF+LogisticRegression、选样、以真实标签模拟专家回复。对比策略共享固定初始集、测试集、复核预算。注意：此为**Oracle离线仿真**，不是真实的LLM在线调用或专家标注系统；在实际无标签专利库中应使用 `select_batch` 和 `export_review_queue` 导出复核队列，再接入独立的人机核验流程。

输出：`active_learning_metrics.csv`（各轮Precision/Recall/F1和复核数量）、`active_learning_selected.json`（选中行索引）。真实应用应将行索引映射为专利ID，并保存每轮版本。

## 3. 蒸馏 `distillation.py`

教师结果以CSV导入，至少包含：`patent_id,teacher_probability`；可增加 `teacher_confidence`。原专利数据至少包含 `patent_id,label,abstract`，标题、权利要求可选。未知人工标签留空，至少30条专家真值样本、正负类皆有。

**教师概率必须来自未见过同一专利真实标签的模型（交叉拟合或独立教师），且不得利用测试集标签生成。** 程序不能自动检测外部教师是否泄漏数据。合成数据 `create_teacher_demo.py` 根据真值生成教师概率，严格仅用于烟雾测试；绝不能作为正式研究教师。

四组实验：
1. `supervised`：仅专家锚点监督训练；
2. `soft_distill`：几乎完全依据教师概率训练；
3. `anchored_distill`：专家硬标签与教师概率加权融合，默认教师权重0.7；
4. `anchored_low_teacher`：教师权重0.3。

学生为 TF-IDF + LogisticRegression，训练目标为各条数据正负两个副本的概率加权交叉熵，并不是PatentBERT/SciBERT神经表征蒸馏、隐藏层匹配或LLM生成推理轨迹蒸馏。输出 `distillation_metrics.csv`、`distillation_predictions.csv` 和数据分割文件。

当前标签级蒸馏可以用于低成本学生模型基线和蒸馏权重消融，但若要证明“知识蒸馏”作为独立算法创新，仍须将教师概率由完整ECLR生成，并添加BERT类模型、不同蒸馏温度、置信度校准及计算成本比较。

## 4. 科研验证的重要限制

1. `kewspi.py` 的原文Spy版本为**启发式近似复现**，不是论文完整Spy算法。
2. 主动学习离线模拟要求所有样本有真实标签；真实无标签样本需要外部LLM/专家核验。**当前两套新增程序可以独立执行，但尚未做到跨轮回写ECLR参数的完整端到端闭环。**
3. 蒸馏中的教师概率由外部CSV导入，尚未实现自动使用ECLR标签模型或LLM教师，不能称为完整教师—学生端到端训练。
4. 条件化标签融合仍使用近似条件独立与相关源折叠，没有实现稀疏概率依赖势函数。
5. 随机划分需要升级为按专利族、时间和专利局隔离。当前实验只是功能验证。
6. 模型可输出很高分类指标，但样本选择策略、类别先验和标签分布都可能明显偏离真实专利库。真实论文应报告多个随机种子、置信区间、复核成本、PR-AUC与校准。
7. **任何真实API密钥、非公开专利全文和未获授权数据都不要提交到GitHub。**

## 5. 建议后续集成

正式V3首先应在每轮主动复核后，把专家确认标签作为ECLR锚点、重新估计标签函数可靠性；随后让学生模型仅接收样本外ECLR软标签，并比较是否优于不蒸馏、不主动学习、固定来源可靠性的各项基准。研发前应完成自动化训练/验证/测试数据切分和无泄漏检查。

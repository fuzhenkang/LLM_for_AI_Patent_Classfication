# KE-WSPI V3：ECLR—主动学习—专家反馈—蒸馏训练闭环

本阶段新增 `closed_loop.py`，将前期分离的三个模块串接为可运行的迭代流程：

```text
专利池与初始人工锚点
   ↓
规则/外部LLM弱标签 L
   ↓
ECLR：基于专家锚点的EM概率融合 q
   ↓
可信度加权软标签训练 TF-IDF + LogisticRegression 学生
   ↓
学生不确定性 + 规则标签冲突 + 文本新颖性
   ↓
预算约束选择复核专利 → 导出 review_queue_round_*.csv
   ├─ oracle：仅在离线模拟中读取已隔离的真实标签
   └─ review：只接受外部审核CSV的明确专家标签
   ↓
新确认标签作为锚点 → 重新估计ECLR → 下一轮
```

## 安装与测试

从 `KE-WSPI` 目录执行：

```bash
pip install -r requirements.txt
python -m pytest -q
python examples/create_demo.py
python closed_loop.py --data examples/demo.csv --mode oracle --out results/closed_demo --rounds 3 --budget 10 --initial 20
```

**合成示例仅证明代码运行，不代表真实研究性能。** 示例中标签由模板直接生成；必须在真实专家标注数据和时间/专利族分割上重新实验。

## 真实无标签专利的复核模式

`patents.csv` 需要 `patent_id,title,abstract,claims,cpc,label`，其中 `label` 允许为空。至少需要24条已确认正负人工标签作为锚点与独立测试数据。启动：

```bash
python closed_loop.py --data patents.csv --mode review --out results/review --rounds 1 --budget 50 --initial 20
```

系统生成 `review_queue_round_1.csv`，包含待审核专利、分类概率、优先度，以及空的 `expert_label` 等字段。审查员填写0/1以及证据后，将其另存为UTF-8 CSV：

```csv
patent_id,expert_label
P001,1
P002,0
```

然后以相同输入和随机种子重运行，并提供专家答复：

```bash
python closed_loop.py --data patents.csv --mode review --reviews completed_reviews.csv --out results/review_updated --rounds 2 --budget 50 --initial 20
```

注意：这一机制是**确定性重运行、重建并消费选中ID的复核记录**，而不是后台运行的持续服务；当前不提供数据库式持久增量状态管理。只对当轮已选中的ID接受专家回写，不会把其他行的预先标注偷偷纳入。

如果已有外部RAG/LLM结果：

```bash
python closed_loop.py --data patents.csv --mode review --llm-jsonl llm_verified.jsonl --out results/review_llm
```

格式遵从主 `README.md`，仅核验原文引文是否存在；无法凭此证明LLM技术判断在语义上正确。

## 输出文件

- `closed_loop_metrics.csv`：每轮专家锚点规模、累计查询数、Precision、Recall、F1、PR-AUC。
- `review_queue_round_1.csv` 等：待核验文本、模型概率、优先分数、审核输入字段。
- `student_round_0.csv` 等：当前学生模型对专利池的预测。
- `split_ids.json`：最初专家锚点、开发、测试与未答复的专利ID。

## 方法边界

1. **真正完成的闭环**是：专家标签更新 → ECLR重估 → 概率标签生成 → 学生加权训练 → 下一轮样本选择；支持离线Oracle实验和外部审查文件反馈。
2. 模型学生使用 TF-IDF + LogisticRegression 的概率软标签蒸馏，**并非神经网络隐藏层/温度蒸馏，也非大模型权重蒸馏**。
3. 未提供自动调用LLM API与向量数据库RAG的实时复核服务；可接收外部LLM的结果文件。
4. ECLR仅具有条件化可靠性及独立来源近似，不包含充分学习的稀疏成对来源依赖势函数。
5. 当前采用分层随机行划分，真实论文需新增专利族隔离、时间外验证、多随机种子重复及置信区间。
6. 在review模式中，候选池里未标注专利无法进入离线测试集；评价仅反映已有人工标注测试集上的表现。
7. 为保护评估，**独立测试集中的专家标签不会被ECLR、复核选样或学生训练使用**。
8. 训练循环中选择分数以学生概率为主；实际研究应比较不同权重与预算，并补充标签纠错率和覆盖率—错误率曲线。
9. 所有合成教师、合成标签和Oracle标签只能用于功能测试，不能成为方法优于基准的科研证据。

## 下阶段可选的生产级扩展

- 在循环中接入真实大模型API提供者、检索知识库和人工双人审核台。
- 加入持久任务数据库、单轮命令与断点恢复。
- 使用PatentBERT/SciBERT替换当前TF-IDF学生。
- 对每轮新增专家标注实施概率抽样权重及来源依赖参数优化。
- 在相同预算与样本覆盖率下实现原文Spy方法的严格复现。

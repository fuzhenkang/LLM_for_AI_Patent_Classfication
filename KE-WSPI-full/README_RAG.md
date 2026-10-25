# KE-WSPI 可配置混合知识检索：中文完整说明

本模块替换先前的JSON + TF-IDF单一路径，保留本地知识原文及ID，新增 **BM25稀疏检索、Embedding + FAISS稠密检索、倒数排序融合（RRF）、可选Cross-Encoder重排序、FAISS持久化索引**。既可独立检索，也可被原有 `run.py → LLMVerifier` 自动调用，不改变ECLR、主动学习、神经蒸馏代码。

## 1. 依赖与模型

依赖已加入 `requirements.txt`：`rank-bm25`、`sentence-transformers`、`faiss-cpu`。默认英文技术知识使用 `BAAI/bge-small-en-v1.5` 向量模型，可修改为可由SentenceTransformers加载的本地目录或Hugging Face模型ID。中英混合知识优先使用经相应语料评估的多语言Embedding模型。

初次使用 `dense` 或 `hybrid` 会下载向量模型并对知识库编码，若启用Reranker，还会额外下载重排序模型；需要联网/模型权重。索引保存为 `index_dir/knowledge_<内容哈希>.faiss`，相同知识内容和Embedding模型名会复用索引。修改知识后生成新索引，不会覆盖旧文件。

## 2. 选择检索模式

修改 `example_config.json` 顶层的 `retrieval`：

```json
"retrieval": {
  "mode": "hybrid",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "index_dir": "cache/rag_index",
  "top_k": 4,
  "recall_k": 12,
  "rrf_k": 60,
  "reranker": {
    "enabled": false,
    "model": "BAAI/bge-reranker-base"
  }
}
```

- `mode: "bm25"`：仅BM25，不需下载Embedding模型，适合基准与低资源环境。
- `mode: "dense"`：仅Embedding + FAISS，相似性为归一化向量点积（余弦相似度）。
- `mode: "hybrid"`：同时召回BM25与FAISS，通过 `sum(1/(rrf_k+rank))` 融合候选排序。
- `reranker.enabled: true`：在召回候选集上使用Cross-Encoder对查询与候选知识对重新打分；默认关闭，以减小开销。
- `top_k`：送入LLM的最终知识条数。
- `recall_k`：每一路检索候选条数。一般应大于或等于top_k。
- `index_dir`：FAISS本地缓存目录。

> 注意：`retrieval.mode="hybrid"` 默认会下载向量模型，但 `llm.enabled=false` 时仍可单独初始化检索器开展检索评估；统一入口 `run.py` 依然构造检索器，故首次运行需要向量模型可用。仅需离线快速验证时，可切换为 `bm25`。

## 3. 单独使用

```python
from hybrid_retrieval import HybridKnowledgeRetriever

retriever = HybridKnowledgeRetriever(
    "examples/knowledge.json",
    {
      "mode": "hybrid",
      "embedding_model": "BAAI/bge-small-en-v1.5",
      "index_dir": "cache/rag_index",
      "top_k": 4,
      "reranker": {"enabled": False}
    }
)
for item in retriever.search("convolutional neural network detects defects"):
    print(item["id"], item["retrieval_score"], item["definition"])
```

## 4. 通过LLM自动使用

`run.py` 已经将 `config.get("retrieval")` 注入 `LLMVerifier`。其执行顺序为：

专利标题/摘要/权利要求 → BM25/FAISS双路召回 → RRF → 可选重排序 → Top-K领域知识 + 专利文本 → Chat Completions兼容LLM API → 证据字符串核验 → LLM弱标签进入ECLR。

`KnowledgeRetriever` 旧类继续留在 `rag_llm.py` 以便历史代码查看，但 `LLMVerifier` 已正式调用 `HybridKnowledgeRetriever`。

## 5. RAG消融与检索评估

建议在相同知识库、相同LLM、相同训练/验证/测试划分、相同调用预算下运行：

| 条件 | 配置 |
|---|---|
| No RAG | 应另设不检索的空证据对照（目前尚未提供） |
| BM25 | `mode=bm25` |
| Dense | `mode=dense` |
| Hybrid | `mode=hybrid` |
| Hybrid + Reranker | `mode=hybrid; reranker.enabled=true` |

检索质量可用知识ID标注集计算 Recall@K、MRR、nDCG@K；下游需同时评价分类 Precision、Recall、F1、PR-AUC、LLM调用量与单件处理成本。

当前实现**不会自动生成知识标注真值**，也未实现向量服务Qdrant或多用户索引更新后台；FAISS仅是本地向量索引，原始JSON仍保存可追溯的知识定义与元数据。

## 6. 数据和泄漏控制

知识JSON每条至少包括唯一的`id`、`title`、`definition`。可额外包含`source`、`version`、`technology_type`、`related_cpc`、`label`等字段。专家案例入库必须来自训练折；独立测试专利及其人工标签绝不能进入案例知识库。时间外实验应使用相应训练时点前可获得的知识版本。

## 7. 已知约束

1. 目前支持基于JSON整体重建索引，不支持大规模增量FAISS索引的复杂同步。
2. BM25仅按空格和字母数字等通用模式切词，中文长段落需要额外中文分词器。
3. 检索评分是相关性排名，不是可直接当成知识真实性或分类概率的校准分数。
4. Cross-Encoder提升与否必须通过实际检索标注及分类任务验证。
5. 在线API调用、模型权重下载与GPU资源需使用者自行提供；不得上传密钥或受限制专利文本至公共仓库。

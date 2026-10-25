"""Configurable persistent hybrid RAG: BM25 + SentenceTransformer embeddings + FAISS + RRF + optional CrossEncoder.
Dependencies for dense mode: sentence-transformers, faiss-cpu, rank-bm25.
"""
import hashlib,json
from pathlib import Path
import numpy as np

def tokenize(text):
    import re
    return re.findall(r"(?u)\b\w+\b",str(text).lower())

class HybridKnowledgeRetriever:
    def __init__(self,knowledge_path,config=None):
        self.config=config or {}
        self.mode=self.config.get("mode","hybrid")
        if self.mode not in ("bm25","dense","hybrid"):raise ValueError("mode must be bm25, dense or hybrid")
        self.docs=json.loads(Path(knowledge_path).read_text(encoding="utf-8"))
        if not isinstance(self.docs,list) or not self.docs:raise ValueError("Knowledge must be a nonempty JSON list")
        self.texts=[str(d.get("title",""))+" "+str(d.get("definition","")) for d in self.docs]
        self.ids=[str(d["id"]) for d in self.docs]
        if len(set(self.ids))!=len(self.ids):raise ValueError("Duplicate knowledge ids")
        self.top_k=int(self.config.get("top_k",4))
        self.recall_k=int(self.config.get("recall_k",12))
        if self.top_k<=0 or self.recall_k<=0:raise ValueError("top_k and recall_k must be positive")
        self.fusion_k=int(self.config.get("rrf_k",60))
        self.index_dir=Path(self.config.get("index_dir","cache/rag_index"))
        self.index_dir.mkdir(parents=True,exist_ok=True)
        self.bm25=None;self.encoder=None;self.index=None;self.reranker=None
        if self.mode in ("bm25","hybrid"):
            from rank_bm25 import BM25Okapi
            self.bm25=BM25Okapi([tokenize(t) or ["__empty__"] for t in self.texts])
        if self.mode in ("dense","hybrid"):
            from sentence_transformers import SentenceTransformer
            import faiss
            self.encoder=SentenceTransformer(self.config.get("embedding_model","BAAI/bge-small-en-v1.5"))
            digest=hashlib.sha256(json.dumps({"ids":self.ids,"texts":self.texts,
                "model":self.config.get("embedding_model","BAAI/bge-small-en-v1.5")},
                ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]
            path=self.index_dir/("knowledge_"+digest+".faiss")
            if path.exists():
                self.index=faiss.read_index(str(path))
                if self.index.ntotal!=len(self.docs):raise ValueError("Cached index size mismatch")
            else:
                vectors=self.encoder.encode(self.texts,convert_to_numpy=True,normalize_embeddings=True,
                    batch_size=int(self.config.get("embedding_batch_size",32)))
                vectors=np.ascontiguousarray(vectors,dtype="float32")
                self.index=faiss.IndexFlatIP(vectors.shape[1]);self.index.add(vectors)
                faiss.write_index(self.index,str(path))
        if self.config.get("reranker",{}).get("enabled",False):
            from sentence_transformers import CrossEncoder
            self.reranker=CrossEncoder(self.config["reranker"].get("model","BAAI/bge-reranker-base"))
    def _bm25_rank(self,query):
        values=self.bm25.get_scores(tokenize(query) or ["__empty__"])
        return np.argsort(-values,kind="stable")[:self.recall_k].tolist(),values
    def _dense_rank(self,query):
        vec=self.encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True)
        scores,idx=self.index.search(np.ascontiguousarray(vec,dtype="float32"),min(len(self.docs),self.recall_k))
        return [int(i) for i in idx[0] if i>=0],scores[0]
    def search(self,query,k=None):
        k=self.top_k if k is None else int(k)
        ranks=[]
        if self.bm25 is not None:ranks.append(self._bm25_rank(query)[0])
        if self.index is not None:ranks.append(self._dense_rank(query)[0])
        fused={}
        for ranking in ranks:
            for pos,i in enumerate(ranking):
                fused[i]=fused.get(i,0.)+1/(self.fusion_k+pos+1)
        candidates=sorted(fused,key=lambda i:(-fused[i],self.ids[i]))[:max(k,self.recall_k)]
        if self.reranker is not None and candidates:
            scores=self.reranker.predict([(query,self.texts[i]) for i in candidates])
            candidates=[candidates[i] for i in np.argsort(-np.asarray(scores),kind="stable")]
        return [{**self.docs[i],"retrieval_score":float(fused[i]),"retrieval_mode":self.mode}
                for i in candidates[:k]]

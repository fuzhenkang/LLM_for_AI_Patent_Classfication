import json
from pathlib import Path
import numpy as np
from hybrid_retrieval import HybridKnowledgeRetriever

def docs(tmp_path):
    path=tmp_path/"knowledge.json"
    path.write_text(json.dumps([
      {"id":"ai","title":"Neural network","definition":"Machine learning neural network classification"},
      {"id":"power","title":"Electrical transformer","definition":"Electromagnetic winding voltage electric transformer"},
      {"id":"edge","title":"Boundary","definition":"Prior art may mention artificial intelligence without using it"}
    ]),encoding="utf-8")
    return path

def test_bm25(tmp_path):
    r=HybridKnowledgeRetriever(docs(tmp_path),{"mode":"bm25","index_dir":str(tmp_path)})
    x=r.search("Neural network classification",2)
    assert len(x)==2 and x[0]["id"]=="ai"

def test_invalid_mode(tmp_path):
    try: HybridKnowledgeRetriever(docs(tmp_path),{"mode":"other"})
    except ValueError:return
    assert False,"invalid retrieval mode must raise"

def test_dense_and_hybrid_with_mock_encoder(tmp_path,monkeypatch):
    import sys,types
    class Encoder:
        def __init__(self,*args,**kwargs):pass
        def encode(self,texts,**kwargs):
            out=[]
            for t in texts:
                s=t.lower()
                if "neural" in s: v=[1.,0.,0.]
                elif "electrical" in s: v=[0.,1.,0.]
                else: v=[0.,0.,1.]
                out.append(v)
            return np.asarray(out,dtype=np.float32)
    fake=types.SimpleNamespace(SentenceTransformer=Encoder,CrossEncoder=object)
    monkeypatch.setitem(sys.modules,"sentence_transformers",fake)
    import pytest
    pytest.importorskip("faiss")
    path=docs(tmp_path)
    for mode in ("dense","hybrid"):
        cfg={"mode":mode,"embedding_model":"mock","index_dir":str(tmp_path/mode),"top_k":2}
        r=HybridKnowledgeRetriever(path,cfg)
        assert r.search("neural model")[0]["id"]=="ai"
        r2=HybridKnowledgeRetriever(path,cfg)
        assert r2.index.ntotal==3

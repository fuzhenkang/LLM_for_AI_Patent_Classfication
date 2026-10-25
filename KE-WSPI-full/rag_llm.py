"""RAG evidence retrieval, OpenAI-compatible LLM interface, verifiable quotations and rule label functions."""
import os,json,re,time,hashlib
from pathlib import Path
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def patent_text(row):
    return " ".join(str(row.get(k,"") or "") for k in ("title","abstract","claims"))

def weak_labels(df):
    """Values: positive=1, negative=-1, abstain=0. CPC absence does not imply negative."""
    a=np.zeros((len(df),4),dtype=np.int64)
    for i,(_,r) in enumerate(df.iterrows()):
        c=str(r.get("cpc","")).upper()
        s=patent_text(r).lower()
        a[i,0]=1 if "G06N" in c else 0
        a[i,1]=1 if re.search(r"neural network|machine learning|deep learning|reinforcement learning",s) else 0
        a[i,2]=-1 if re.search(r"electrical transformer|transformer winding|magnetic flux",s) else 0
        a[i,3]=1 if re.search(r"trained (?:classifier|model)|learning algorithm",s) else 0
    return a

class KnowledgeRetriever:
    """Local reproducible hybrid TF-IDF retrieval (word + char); no external hosted search."""
    def __init__(self,path):
        self.docs=json.loads(Path(path).read_text(encoding="utf-8"))
        if not self.docs: raise ValueError("Empty knowledge base")
        corpus=[d["title"]+" "+d["definition"] for d in self.docs]
        self.w=TfidfVectorizer(ngram_range=(1,2),stop_words="english").fit(corpus)
        self.c=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5)).fit(corpus)
        self.W=self.w.transform(corpus);self.C=self.c.transform(corpus)
    def search(self,text,k=4):
        z=.6*cosine_similarity(self.w.transform([text]),self.W)[0]+.4*cosine_similarity(self.c.transform([text]),self.C)[0]
        idx=np.argsort(-z)[:k]
        return [{**self.docs[int(i)],"retrieval_score":float(z[i])} for i in idx]

def quote_supported(quote,source):
    norm=lambda z:" ".join(str(z).lower().split())
    return bool(quote) and norm(quote) in norm(source)

class LLMVerifier:
    def __init__(self,config,knowledge_path,cache_dir="cache/llm",retrieval_config=None):
        self.cfg=config
        from hybrid_retrieval import HybridKnowledgeRetriever
        self.rag=HybridKnowledgeRetriever(knowledge_path,retrieval_config or {"mode":"bm25"})
        self.path=Path(cache_dir);self.path.mkdir(parents=True,exist_ok=True)
    def verify(self,patent):
        pid=str(patent["patent_id"]);text=patent_text(patent)
        evidence=self.rag.search(text)
        if not self.cfg.get("enabled",False):
            return {"patent_id":pid,"label":0,"confidence":0,"evidence":[],"status":"disabled"}
        payload={"model":self.cfg["model"],"temperature":0,"response_format":{"type":"json_object"},
          "messages":[
           {"role":"system","content":"You are a patent technology annotator. Return JSON with label 1=AI,-1=non-AI,0=abstain, confidence in [0,1], evidence list of EXACT VERBATIM quotes from provided PATENT text. No invented quotes. Technical AI use matters; background-only mention is insufficient. Respond only JSON."},
           {"role":"user","content":json.dumps({"patent_id":pid,"patent_text":text[:14000],"retrieved_knowledge":evidence},ensure_ascii=False)}
          ]}
        key=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        f=self.path/(key+".json")
        if f.exists():return json.loads(f.read_text(encoding="utf-8"))
        token=os.environ.get(self.cfg.get("api_key_env","OPENAI_API_KEY"))
        if not token:raise EnvironmentError("Missing LLM API key environment variable")
        endpoint=self.cfg["base_url"].rstrip("/")+"/chat/completions"
        last=None
        for t in range(3):
            try:
                response=requests.post(endpoint,headers={"Authorization":"Bearer "+token},
                    json=payload,timeout=self.cfg.get("timeout",90))
                response.raise_for_status()
                raw=response.json()["choices"][0]["message"]["content"]
                obj=json.loads(raw)
                label=int(obj.get("label",0))
                if label not in (-1,0,1):label=0
                quotes=obj.get("evidence",[])
                if isinstance(quotes,str):quotes=[quotes]
                quotes=[q if isinstance(q,str) else q.get("quote","") for q in quotes if isinstance(q,(str,dict))]
                correct=[q for q in quotes if quote_supported(q,text)]
                good=bool(correct) and len(correct)==len(quotes)
                result={"patent_id":pid,"label":label if good else 0,
                  "confidence":float(np.clip(obj.get("confidence",0),0,1)) if good else 0.,
                  "evidence":correct,"status":"verified" if good else "unsupported",
                  "knowledge_ids":[k["id"] for k in evidence]}
                f.write_text(json.dumps(result,ensure_ascii=False),encoding="utf-8")
                return result
            except (requests.RequestException,KeyError,ValueError,TypeError) as err:
                last=err
                time.sleep(.5*(t+1))
        return {"patent_id":pid,"label":0,"confidence":0,"evidence":[],"status":"error: "+str(last)[:200]}

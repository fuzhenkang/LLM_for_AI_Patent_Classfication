"""KE-WSPI minimal executable evidence-aware weak supervision pipeline."""
import argparse, json, re
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split

def weak_functions(df, terms=("machine learning","neural network","deep learning")):
    out=np.zeros((len(df),3),dtype=int)
    for i,r in enumerate(df.to_dict("records")):
        t=" ".join(str(r.get(k,"")) for k in ("title","abstract","claims")).lower()
        out[i,0]=1 if "G06N" in str(r.get("cpc","")).upper() else 0
        out[i,1]=1 if any(x in t for x in terms) else 0
        out[i,2]=-1 if any(x in t for x in ("power transformer","transformer coil")) else 0
    return out

def evidence_lf(df, jsonl):
    replies={str(x["patent_id"]):x for x in (json.loads(line) for line in open(jsonl,encoding="utf-8"))}
    labels=np.zeros(len(df),dtype=int);scores=np.zeros(len(df))
    for i,r in enumerate(df.to_dict("records")):
        x=replies.get(str(r["patent_id"]),{})
        txt=" ".join(str(r.get(k,"")) for k in ("title","abstract","claims")).lower()
        quotes=[e.get("quote","") for e in x.get("evidence",[]) if isinstance(e,dict)]
        scores[i]=float(bool(quotes) and all(q and " ".join(q.lower().split()) in " ".join(txt.split()) for q in quotes))
        if scores[i] and x.get("label") in (0,1): labels[i]=1 if x["label"]==1 else -1
    return labels,scores

class ECLR:
    """Expert-anchored EM; group-shrinkage and optional binary evidence reliability.
    Conditional independence is assumed; duplicate source collapsing approximates dependence.
    """
    def __init__(self, group_mode=False, evidence_mode=False, iterations=15):
        self.group_mode,self.evidence_mode,self.iterations=group_mode,evidence_mode,iterations

    def fit(self,L,anchors,y,groups=None,evidence=None):
        L=np.asarray(L,dtype=int); self.m=L.shape[1]
        self.g=np.asarray(groups if self.group_mode else np.zeros(len(L)),dtype=int)
        self.e=(np.asarray(evidence)>=1).astype(int) if self.evidence_mode else np.zeros(len(L),dtype=int)
        self.ng=self.g.max()+1;self.ne=2 if self.evidence_mode else 1
        idx=L+1;anchors=np.asarray(anchors,dtype=int);y=np.asarray(y,dtype=int)
        if not len(anchors) or len(np.unique(y))<2: raise ValueError("Two anchor classes required")
        q=np.full(len(L),.15);q[anchors]=y
        for _ in range(self.iterations):
            global_counts=np.ones((2,self.m,3))
            for c in (0,1):
                w=q if c else 1-q
                for j in range(self.m):global_counts[c,j]+=np.bincount(idx[:,j],weights=w,minlength=3)
            gp=global_counts/global_counts.sum(axis=-1,keepdims=True)
            self.params=np.empty((self.ng,self.ne,2,self.m,3))
            for g in range(self.ng):
                for e in range(self.ne):
                    for c in (0,1):
                        w=q if c else 1-q
                        for j in range(self.m):
                            mask=(self.g==g)&((self.e==e) if self.evidence_mode and j==self.m-1 else True)
                            counts=np.bincount(idx[mask,j],weights=w[mask],minlength=3)
                            self.params[g,e,c,j]=(counts+25*gp[c,j])/(counts.sum()+25)
            new=self.predict(L,groups,evidence)
            q=new;q[anchors]=y
        return self

    def predict(self,L,groups=None,evidence=None):
        L=np.asarray(L,dtype=int);idx=L+1
        g=np.asarray(groups if self.group_mode else np.zeros(len(L)),dtype=int)
        e=(np.asarray(evidence)>=1).astype(int) if self.evidence_mode else np.zeros(len(L),dtype=int)
        logits=np.full(len(L),np.log(.15/.85))
        for j in range(L.shape[1]):
            logits+=np.log(self.params[g,e,1,j,idx[:,j]])-np.log(self.params[g,e,0,j,idx[:,j]])
        return expit(logits)

def collapse(L):
    # CPC + keyword grouped to avoid counting correlated positives twice
    a=L[:,:2];c=np.where((a==1).any(1)&~(a==-1).any(1),1,np.where((a==-1).any(1)&~(a==1).any(1),-1,0))
    return np.column_stack((c,L[:,2:]))

def spy_proxy(L,X,seed=42):
    """Spy-inspired baseline, NOT exact Wu et al. (2026) implementation."""
    rng=np.random.default_rng(seed)
    pos=np.where((L==1).any(1)&~(L==-1).any(1))[0]
    neg=np.where((L==-1).any(1)&~(L==1).any(1))[0]
    if len(pos)<4 or len(neg)<4:raise ValueError("Insufficient Spy candidates")
    spies=rng.choice(neg,size=max(1,len(neg)//4),replace=False)
    ids=np.r_[pos,neg];y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    y[np.isin(ids,spies)]=1
    m=LogisticRegression(max_iter=400).fit(X[ids],y)
    p=m.predict_proba(X)[:,1]
    keep_pos=pos[p[pos]>np.quantile(p[spies],.85)]
    if len(keep_pos)<4:return np.zeros(len(L),dtype=int)
    spies2=rng.choice(keep_pos,size=max(1,len(keep_pos)//4),replace=False)
    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))];y[np.isin(ids,spies2)]=0
    m=LogisticRegression(max_iter=400).fit(X[ids],y);p=m.predict_proba(X)[:,1]
    keep_neg=neg[p[neg]<np.quantile(p[spies2],.01)]
    result=np.zeros(len(L),dtype=int);result[keep_pos]=1;result[keep_neg]=-1
    return result

def evaluate(df,out,llm=None,seed=42):
    from pathlib import Path
    df=df.copy().fillna({"title":"","abstract":"","claims":"","cpc":""})
    if df.patent_id.duplicated().any():raise ValueError("Duplicate patent ID")
    gold=np.where(df.label.notna())[0];gold_y=df.label.to_numpy()
    if len(gold)<30 or len(np.unique(gold_y[gold]))<2:raise ValueError("Need >=30 expert labels of both classes")
    anchor,hold=train_test_split(gold,test_size=.4,stratify=gold_y[gold],random_state=seed)
    dev,test=train_test_split(hold,test_size=.5,stratify=gold_y[hold],random_state=seed)
    train=np.array(sorted(set(range(len(df)))-set(dev)-set(test)))
    txt=df[["title","abstract","claims"]].astype(str).agg(" ".join,axis=1).tolist()
    tf=TfidfVectorizer(max_features=12000,ngram_range=(1,2))
    tf.fit([txt[i] for i in train]);X=tf.transform(txt)
    L=weak_functions(df)
    el=np.zeros(len(df));es=np.zeros(len(df))
    if llm:el,es=evidence_lf(df,llm)
    L=np.column_stack((L,el))
    groups=pd.factorize(df.cpc.astype(str).str[:3])[0]
    results=[];predictions=[]
    for mode in ("majority","fixed","group","evidence","collapsed","spy_proxy"):
        selected=train.copy()
        try:
            if mode=="spy_proxy":
                labels=spy_proxy(L[train,:3],X[train],seed)
                selected=train[labels!=0];p_train=(labels[labels!=0]==1).astype(float)
            elif mode=="majority":
                votes=L[train].sum(axis=1);selected=train[votes!=0];p_train=(votes[votes!=0]>0).astype(float)
            else:
                local=collapse(L) if mode=="collapsed" else L
                g_mode=mode in ("group","evidence","collapsed")
                e_mode=mode in ("evidence","collapsed")
                anchor_local=np.searchsorted(train,anchor)
                model=ECLR(g_mode,e_mode).fit(local[train],anchor_local,gold_y[anchor].astype(int),
                                                groups[train] if g_mode else None,es[train] if e_mode else None)
                p_train=model.predict(local[train],groups[train] if g_mode else None,es[train] if e_mode else None)
            if len(selected)<5:raise ValueError("Insufficient samples")
            student=LogisticRegression(max_iter=500,random_state=seed)
            if mode not in ("majority","spy_proxy"):
                w=np.clip(p_train,.001,.999)
                expanded=np.r_[selected,selected]
                yy=np.r_[np.ones(len(selected)),np.zeros(len(selected))]
                weights=np.r_[w,1-w]
                anchored=np.isin(expanded,anchor)
                weights[anchored]=np.where(yy[anchored]==gold_y[expanded[anchored]],1.,0.)
                student.fit(X[expanded],yy,sample_weight=weights)
            else:
                student.fit(X[selected],p_train)
            prob=student.predict_proba(X[test])[:,1]
            yt=gold_y[test].astype(int); yp=(prob>=.5).astype(int)
            row={"variant":mode,"status":"ok","train_size":len(selected),
                 "precision":precision_score(yt,yp,zero_division=0),"recall":recall_score(yt,yp,zero_division=0),
                 "f1":f1_score(yt,yp,zero_division=0),"pr_auc":average_precision_score(yt,prob),"auroc":roc_auc_score(yt,prob)}
            predictions.extend({"patent_id":str(df.iloc[i].patent_id),"variant":mode,"gold":int(yt[k]),"probability":float(prob[k])} for k,i in enumerate(test))
        except (ValueError,IndexError) as error:row={"variant":mode,"status":"skipped: "+str(error)}
        results.append(row)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(predictions).to_csv(out/"predictions.csv",index=False)
    (out/"splits.json").write_text(json.dumps({"anchor":df.iloc[anchor].patent_id.tolist(),"dev":df.iloc[dev].patent_id.tolist(),"test":df.iloc[test].patent_id.tolist()},ensure_ascii=False,indent=2),encoding="utf-8")
    return results

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("--llm-jsonl");p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    print(json.dumps(evaluate(pd.read_csv(a.data),a.out,a.llm_jsonl,a.seed),ensure_ascii=False,indent=2))

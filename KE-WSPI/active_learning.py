"""主动学习：预算约束选择、不确定性/分歧/新颖性、复核队列与离线学习曲线。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def score_candidates(probabilities, votes=None, features=None, strategy="hybrid", reviewed=None):
    """打分只使用未标注特征和模型输出；reviewed是已获真实复核的行位置。"""
    p=np.clip(np.asarray(probabilities,dtype=float),1e-8,1-1e-8)
    entropy=-(p*np.log(p)+(1-p)*np.log(1-p))/np.log(2)
    disagreement=np.zeros(len(p))
    if votes is not None:
        v=np.asarray(votes)
        if v.ndim!=2 or v.shape[0]!=len(p): raise ValueError("votes shape mismatch")
        nonzero=(v!=0).sum(axis=1)
        disagreement=np.divide(np.minimum((v>0).sum(1),(v<0).sum(1)),nonzero,out=np.zeros(len(p)),where=nonzero>0)*2
    novelty=np.zeros(len(p))
    if features is not None and reviewed is not None and len(reviewed):
        # normalized TF-IDF -> cosine similarity
        r=features[reviewed]
        sims=features@r.T
        sims=sims.toarray() if issparse(sims) else np.asarray(sims)
        novelty=1-sims.max(axis=1)
    if strategy=="entropy": score=entropy
    elif strategy=="disagreement": score=disagreement
    elif strategy=="novelty": score=novelty
    elif strategy=="hybrid": score=.5*entropy+.3*disagreement+.2*novelty
    elif strategy=="random": score=np.zeros(len(p))
    else: raise ValueError("Unknown strategy: "+strategy)
    return score

def select_batch(probabilities,budget,eligible,votes=None,features=None,reviewed=None,
                 strategy="hybrid",seed=42):
    eligible=np.asarray(eligible,dtype=int)
    if budget<=0 or len(eligible)==0:return np.array([],dtype=int)
    rng=np.random.default_rng(seed)
    if strategy=="random":return rng.choice(eligible,size=min(budget,len(eligible)),replace=False)
    score=score_candidates(probabilities,votes,features,strategy,reviewed)
    # stable random tie breaking
    ranked=eligible[np.lexsort((rng.random(len(eligible)),-score[eligible]))]
    return ranked[:min(budget,len(ranked))]

def export_review_queue(df,selected,probabilities,scores,out):
    wanted=df.iloc[selected].copy()
    wanted["model_probability"]=np.asarray(probabilities)[selected]
    wanted["review_priority"]=np.asarray(scores)[selected]
    wanted["expert_label"]=""
    wanted["evidence_quote"]=""
    wanted["reviewer_id"]=""
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    wanted.to_csv(path,index=False,encoding="utf-8-sig")
    return path

def offline_learning_curve(df,strategies=("random","entropy","disagreement","hybrid"),
                           initial=20,batch=10,rounds=5,seed=42):
    """仅用于全部样本拥有专家真值的离线模拟。test永不进入查询池。"""
    if df["label"].isna().any(): raise ValueError("Offline simulation needs labels for all rows")
    y=df["label"].astype(int).to_numpy()
    if set(np.unique(y))!={0,1}:raise ValueError("Need binary 0/1 labels")
    ids=np.arange(len(df))
    pool,test=train_test_split(ids,test_size=.25,random_state=seed,stratify=y)
    initial=max(initial,4)
    rng=np.random.default_rng(seed)
    c0=pool[y[pool]==0];c1=pool[y[pool]==1]
    n0=max(2,initial//2);n1=max(2,initial-n0)
    if len(c0)<n0 or len(c1)<n1:raise ValueError("Not enough initial labels")
    start=np.r_[rng.choice(c0,n0,replace=False),rng.choice(c1,n1,replace=False)]
    texts=df[["title","abstract","claims"]].fillna("").astype(str).agg(" ".join,axis=1).tolist()
    tf=TfidfVectorizer(max_features=20000,ngram_range=(1,2))
    tf.fit([texts[i] for i in pool]); X=tf.transform(texts)
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location("kewspi_core",str(Path(__file__).with_name("kewspi.py")))
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    votes=mod.weak_functions(df.fillna(""))
    rows=[]; queues={}
    for strategy in strategies:
        reviewed=set(start.tolist())
        for turn in range(rounds+1):
            train=np.array(sorted(reviewed))
            clf=LogisticRegression(max_iter=700,random_state=seed).fit(X[train],y[train])
            p=clf.predict_proba(X)[:,1]
            yp=(p[test]>=.5).astype(int)
            rows.append({"strategy":strategy,"round":turn,"reviewed":len(train),
                         "precision":precision_score(y[test],yp,zero_division=0),
                         "recall":recall_score(y[test],yp,zero_division=0),
                         "f1":f1_score(y[test],yp,zero_division=0)})
            if turn==rounds:break
            eligible=np.array(sorted(set(pool.tolist())-reviewed))
            chosen=select_batch(p,batch,eligible,votes,X,train,strategy,seed+turn)
            queues[(strategy,turn)]=chosen.tolist()
            reviewed.update(chosen.tolist())
    return pd.DataFrame(rows),queues

def main():
    parser=argparse.ArgumentParser(description="KE-WSPI 离线主动学习实验（需全量真值）")
    parser.add_argument("--data",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--initial",type=int,default=20);parser.add_argument("--batch",type=int,default=10)
    parser.add_argument("--rounds",type=int,default=5);parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    df=pd.read_csv(args.data)
    for c in ("title","abstract","claims"):
        if c not in df:df[c]=""
    res,q=offline_learning_curve(df,initial=args.initial,batch=args.batch,rounds=args.rounds,seed=args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    res.to_csv(out/"active_learning_metrics.csv",index=False)
    import json
    (out/"active_learning_selected.json").write_text(json.dumps({str(k):v for k,v in q.items()},ensure_ascii=False,indent=2),encoding="utf-8")
    print(res.to_string(index=False))
if __name__=="__main__":main()

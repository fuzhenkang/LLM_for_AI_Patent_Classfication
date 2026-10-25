"""Unified KE-WSPI end-to-end CLI. All data transformations fit on train only.
LLM calls require configured service credentials; optional expert oracle for synthetic simulation only.
"""
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score,roc_auc_score
from sklearn.model_selection import train_test_split
from rag_llm import LLMVerifier,weak_labels,patent_text
from label_graph import DependentLabelModel
from spy_baseline import spy_refinement

def metrics(y,p):
 pred=(p>=.5).astype(int)
 return dict(precision=float(precision_score(y,pred,zero_division=0)),
  recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
  pr_auc=float(average_precision_score(y,p)),
  roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None)

def train_lightweight(X,q,anchors,anchor_labels,seed):
 """soft-label weighted empirical risk minimization: two copies, opposite labels."""
 from scipy.sparse import vstack
 n=len(q);targets=np.clip(np.asarray(q).copy(),.0001,.9999)
 targets[np.asarray(anchors)]=np.asarray(anchor_labels)
 m=LogisticRegression(max_iter=700,random_state=seed)
 m.fit(vstack([X,X]),np.r_[np.ones(n),np.zeros(n)],
       sample_weight=np.r_[targets,1-targets])
 return m

def run(config):
 seed=int(config.get("seed",42));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 out=Path(config["output"]);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(config["data"],dtype={"patent_id":str})
 for c in ("title","abstract","claims","cpc"):
  if c not in df:df[c]=""
  df[c]=df[c].fillna("").astype(str)
 if not {"patent_id","label"}.issubset(df):raise ValueError("Need patent_id,label fields")
 if df.patent_id.duplicated().any():raise ValueError("Duplicate patent ids")
 gold=pd.to_numeric(df.label,errors="coerce").to_numpy()
 valid=np.flatnonzero(np.isfinite(gold))
 if len(valid)<40 or len(np.unique(gold[valid]))<2:raise ValueError("At least 40 gold examples from 2 classes needed for comparable evaluation")
 known_train,hold=train_test_split(valid,test_size=.4,stratify=gold[valid],random_state=seed)
 dev,test=train_test_split(hold,test_size=.5,stratify=gold[hold],random_state=seed)
 train=np.array(sorted(set(range(len(df)))-set(dev)-set(test)))
 loop=config.get("loop",{})
 ninit=int(loop.get("initial_gold",16))
 rng=np.random.default_rng(seed)
 pos=known_train[gold[known_train]==1];neg=known_train[gold[known_train]==0]
 if len(pos)<2 or len(neg)<2:raise ValueError("Insufficient supervised training anchors")
 anchors=set(np.r_[rng.choice(pos,min(len(pos),max(2,ninit//2)),replace=False),
                    rng.choice(neg,min(len(neg),max(2,ninit//2)),replace=False)].tolist())
 oracle=bool(loop.get("auto_oracle",False))
 if oracle and np.any(~np.isfinite(gold[train])):raise ValueError("auto_oracle only permitted for all-labeled synthetic/offline data")
 text=df[["title","abstract","claims"]].agg(" ".join,axis=1).tolist()
 tf=TfidfVectorizer(max_features=20000,ngram_range=(1,2))
 X=tf.fit_transform([text[i] for i in train]);Xall=tf.transform(text)
 W=weak_labels(df)
 llm=np.zeros(len(df),dtype=np.int64);evidence=np.zeros(len(df))
 verifier=LLMVerifier(config["llm"],config["knowledge"],str(out/"cache"),config.get("retrieval",{"mode":"bm25"}))
 groups=pd.factorize(df.cpc.str.slice(0,3))[0]
 # groups remap to zero-index only for training nodes
 unique,gr=np.unique(groups[train],return_inverse=True)
 group_all=np.zeros(len(df),dtype=np.int64);group_all[train]=gr
 rounds=int(loop.get("rounds",2));budget=int(loop.get("query_budget",8))
 if len(anchors)>len(train):raise ValueError("Invalid anchor split")
 evaluation=[];reviewed=set();student=None
 for t in range(rounds+1):
  anchor=np.array(sorted(anchors),dtype=int);anchor_local=np.searchsorted(train,anchor)
  if not np.array_equal(train[anchor_local],anchor):raise RuntimeError("Anchor leakage")
  # Query LLM on informative patent candidates before fusion. Never touch dev/test.
  if config["llm"].get("enabled",False) and t>0:
   candidates=np.array(sorted(set(train.tolist())-anchors-reviewed),dtype=int)
   count=int(config["llm"].get("max_calls_per_round",budget))
   if student is not None and len(candidates):
    candidates=candidates[np.argsort(np.abs(last_probs[candidates]-.5))[:count]]
   else:candidates=candidates[:count]
   for ix in candidates:
    ans=verifier.verify(df.iloc[ix].to_dict())
    llm[ix]=int(ans["label"]);evidence[ix]=float(ans["confidence"])
  labels=np.column_stack([W,llm])
  # Ablation stages share same teacher training anchors and same held-out tests.
  methods={"majority":None,"independent":[],"dependent":[(0,1),(1,3)]}
  preds={}
  for name,pairs in methods.items():
   if name=="majority":
    v=np.sign(labels[train].sum(1))
    q=np.where(v>0,.8,np.where(v<0,.2,.5))
   else:
    model=DependentLabelModel(labels.shape[1],int(gr.max())+1,pairs=pairs)
    model.fit(labels[train],anchor_local,gold[anchor].astype(int),gr,
        epochs=int(config["label_model"].get("epochs",120)),
        lr=float(config["label_model"].get("learning_rate",.025)),
        pair_l2=float(config["label_model"].get("pair_l2",.2)),seed=seed)
    q=model.predict(labels[train],gr)
   light=train_lightweight(X,q,anchor_local,gold[anchor].astype(int),seed)
   p=light.predict_proba(Xall[test])[:,1]
   preds[name]=p
   evaluation.append({"round":t,"variant":name,**metrics(gold[test].astype(int),p),
                     "gold_train":len(anchor),"llm_calls":int((llm!=0).sum())})
   if name=="dependent":
    last_probs=light.predict_proba(Xall)[:,1]
    student=light;teacher_q=q
  # Spy comparator uses training only; student model evaluated on same test
  positive=train[(W[train]==1).any(1)&~(W[train]==-1).any(1)]
  negative=train[(W[train]==-1).any(1)&~(W[train]==1).any(1)]
  if len(positive)>4 and len(negative)>4:
   try:
    spy,info=spy_refinement([text[i] for i in train],
       np.searchsorted(train,positive),np.searchsorted(train,negative),random_state=seed)
    chosen=np.where(spy!=0)[0]
    if len(np.unique(spy[chosen]))==2:
     sm=LogisticRegression(max_iter=700,random_state=seed).fit(X[chosen],(spy[chosen]==1).astype(int))
     p=sm.predict_proba(Xall[test])[:,1]
     evaluation.append({"round":t,"variant":"spy_reconstruction",**metrics(gold[test].astype(int),p),
       "gold_train":len(anchor),"llm_calls":0,"spy_note":info["original_manual_CPC_filter"]})
   except ValueError:pass
  (out/f"round_{t}_teacher.csv").write_text(pd.DataFrame({"patent_id":df.iloc[train].patent_id,
        "probability":teacher_q}).to_csv(index=False),encoding="utf-8")
  if t==rounds:break
  candidate=np.array(sorted(set(train.tolist())-anchors-reviewed),dtype=int)
  if not len(candidate):break
  # entropy + disagreement (with weak voting) + rarity within current train set
  uncertainty=1-2*np.abs(last_probs[candidate]-.5)
  vote=labels[candidate]
  diss=np.divide(np.minimum((vote>0).sum(1),(vote<0).sum(1)),np.maximum((vote!=0).sum(1),1))*2
  scores=.7*uncertainty+.3*diss
  chosen=candidate[np.argsort(-scores)[:budget]]
  query=df.iloc[chosen].copy();query["student_probability"]=last_probs[chosen]
  query["expert_label"]=""
  query.to_csv(out/f"review_round_{t+1}.csv",index=False)
  reviewed.update(map(int,chosen))
  if oracle:anchors.update(map(int,chosen))
  else:
   # The single invocation terminates without fabricating expert labels.
   # Human review CSV can be merged into the main label column on the next invocation.
   break
  # Neural student training optional; costly model downloads require configured environment.
  if config.get("neural_enabled",False):
   from neural_distill import train_student
   labels_train=np.full(len(train),np.nan)
   labels_train[anchor_local]=gold[anchor]
   train_student(df.iloc[train].reset_index(drop=True),teacher_q,
          config["student"],out/f"neural_round_{t}",expert_labels=labels_train)
 # Ensure neural model can be trained even with zero active-learning rounds
 if config.get("neural_enabled",False) and rounds==0:
  from neural_distill import train_student
  anchor=np.array(sorted(anchors));loc=np.searchsorted(train,anchor)
  hard=np.full(len(train),np.nan);hard[loc]=gold[anchor]
  train_student(df.iloc[train].reset_index(drop=True),teacher_q,
                config["student"],out/"neural_round_0",expert_labels=hard)
 report=pd.DataFrame(evaluation);report.to_csv(out/"experiments.csv",index=False)
 (out/"split.json").write_text(json.dumps({"train":df.iloc[train].patent_id.tolist(),
     "dev":df.iloc[dev].patent_id.tolist(),"test":df.iloc[test].patent_id.tolist()},ensure_ascii=False,indent=2))
 return report

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--config",default="example_config.json")
 ap.add_argument("--neural",action="store_true",help="Enable downloadable transformer student training")
 a=ap.parse_args()
 conf=json.loads(Path(a.config).read_text(encoding="utf-8"))
 if a.neural:conf["neural_enabled"]=True
 print(run(conf).to_string(index=False))

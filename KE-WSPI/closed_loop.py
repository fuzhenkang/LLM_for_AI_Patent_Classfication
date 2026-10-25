"""KE-WSPI integrated closed loop: ECLR -> active querying -> oracle/review -> ECLR update -> soft-label student.
No live LLM in this script. Offline oracle is strictly for simulation, external review CSV for real use.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score
from kewspi import weak_functions,ECLR,evidence_lf
from active_learning import select_batch,export_review_queue
from distillation import fit_student

def run_loop(df,out,mode="oracle",reviews=None,rounds=4,budget=10,seed=42,
             llm_jsonl=None,initial=20):
    """In oracle mode all labels required; held-out gold labels NEVER enter weak model or active selection.
    In review mode, expert labels are stored in 'label'; review CSV is an additional labeled pool,
    optional. Split only expert-labeled data; unknown rows remain available to train/choose.
    """
    df=df.reset_index(drop=True).copy()
    for c in ["title","abstract","claims","cpc"]:
        if c not in df:df[c]=""
        df[c]=df[c].fillna("").astype(str)
    if "patent_id" not in df or df.patent_id.duplicated().any():raise ValueError("unique patent_id required")
    if "label" not in df:df["label"]=np.nan
    y=pd.to_numeric(df.label,errors="coerce").to_numpy()
    known=np.flatnonzero(np.isfinite(y))
    if len(known)<24 or set(np.unique(y[known]))!={0.,1.}:raise ValueError("need >=24 expert gold labels of both classes")\n    if mode=="oracle" and len(known)!=len(df):raise ValueError("oracle mode requires expert labels for every patent; use review for real unlabeled data")
    pool,hold=train_test_split(known,test_size=.4,stratify=y[known],random_state=seed)
    dev,test=train_test_split(hold,test_size=.5,stratify=y[hold],random_state=seed)
    rng=np.random.default_rng(seed)
    c0=pool[y[pool]==0];c1=pool[y[pool]==1]
    k0=max(2,initial//2);k1=max(2,initial-k0)
    if len(c0)<k0 or len(c1)<k1:raise ValueError("not enough pool examples for initial")
    anchors=set(np.r_[rng.choice(c0,k0,replace=False),rng.choice(c1,k1,replace=False)].tolist())\n    initial_anchor_ids=sorted(anchors)
    # In real review mode, labels from CSV are additional human checks; never overwrite held-out gold
    review_map={}
    if reviews:
        rr=pd.read_csv(reviews,dtype={"patent_id":str})
        if not {"patent_id","expert_label"}.issubset(rr):raise ValueError("reviews require patent_id,expert_label")
        review_map={str(row.patent_id):int(row.expert_label) for row in rr.itertuples() if pd.notna(row.expert_label) and str(row.expert_label) in ("0","1","0.0","1.0")}
    texts=df[["title","abstract","claims"]].agg(" ".join,axis=1)
    fit_idx=np.array(sorted(set(range(len(df)))-set(dev)-set(test)))
    vec=TfidfVectorizer(max_features=12000,ngram_range=(1,2)).fit(texts.iloc[fit_idx])
    X=vec.transform(texts)
    lf=weak_functions(df);el=np.zeros(len(df),dtype=int);evidence=np.zeros(len(df))
    if llm_jsonl:el,evidence=evidence_lf(df,llm_jsonl)
    L=np.column_stack((lf,el))
    groups=pd.factorize(df.cpc.str[:3])[0]
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metrics=[];queried=set(); pending=[]
    prior_probs=np.full(len(df),.5)
    for step in range(rounds+1):
        # consume only known reviews selected in PREVIOUS rounds
        for i in list(queried):
            pid=str(df.iloc[i].patent_id)
            if pid in review_map:
                y[i]=review_map[pid];anchors.add(i)
        tr=np.array(sorted(anchors),dtype=int)
        # all non-held-out records are allowed for label modeling; only train anchors expose labels
        anchor_positions=np.searchsorted(fit_idx,tr)
        if not np.all(fit_idx[anchor_positions]==tr): raise RuntimeError("anchor lookup failed")
        label_model=ECLR(group_mode=True,evidence_mode=True,iterations=12)
        label_model.fit(L[fit_idx],anchor_positions,y[tr].astype(int),groups[fit_idx],evidence[fit_idx])
        q=label_model.predict(L[fit_idx],groups[fit_idx],evidence[fit_idx])
        # student uses ECLR probabilities except confirmed expert labels override on fitted anchors
        conf=np.maximum(q,1-q)
        student=fit_student(X[fit_idx],q,confidence=conf,
           hard_ids=anchor_positions,hard_labels=y[tr].astype(int),alpha=.6,seed=seed)
        probs=student.predict_proba(X)[:,1];truth=y[test].astype(int);p=probs[test]
        metrics.append({"round":step,"verified_train":len(anchors),"queried":len(queried),
          "precision":precision_score(truth,p>=.5,zero_division=0),
          "recall":recall_score(truth,p>=.5,zero_division=0),
          "f1":f1_score(truth,p>=.5,zero_division=0),
          "pr_auc":average_precision_score(truth,p)})
        if step==rounds:break
        eligible=np.array(sorted(set(fit_idx)-anchors-queried),dtype=int)
        if not len(eligible):break
        chosen=select_batch(probs,budget,eligible,L,X,tr,strategy="hybrid",seed=seed+step)
        score=np.abs(probs-.5)
        export_review_queue(df,chosen,probs,1-score,out/f"review_queue_round_{step+1}.csv")
        # oracle reveals gold for selected train-pool records only; in real mode awaits supplied reviews
        if mode=="oracle":
            unknown=[i for i in chosen if not np.isfinite(y[i])]
            if unknown:raise ValueError("oracle mode requires gold on every chosen record")
            anchors.update(chosen.tolist())
        elif mode=="review":
            # in review mode, only supplied responses may be added; unanswered ids remain quarantined
            for i in chosen:
                pid=str(df.iloc[i].patent_id)
                if pid in review_map:
                    y[i]=review_map[pid];anchors.add(int(i))
                else:pending.append(pid)
        else:raise ValueError("mode must be oracle or review")
        queried.update(chosen.tolist())
        pd.DataFrame({"patent_id":df.patent_id,"student_probability":probs}).to_csv(out/f"student_round_{step}.csv",index=False)
    pd.DataFrame(metrics).to_csv(out/"closed_loop_metrics.csv",index=False)
    (out/"split_ids.json").write_text(json.dumps(
      {"initial_anchors":df.iloc[initial_anchor_ids].patent_id.tolist(),
       "dev":df.iloc[dev].patent_id.tolist(),"test":df.iloc[test].patent_id.tolist(),
       "pending_review":sorted(set(pending)-{str(df.iloc[i].patent_id) for i in anchors})},
       ensure_ascii=False,indent=2),encoding="utf-8")
    return pd.DataFrame(metrics)

def cli():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--mode",choices=["oracle","review"],default="oracle")
    ap.add_argument("--reviews");ap.add_argument("--llm-jsonl")
    ap.add_argument("--rounds",type=int,default=4);ap.add_argument("--budget",type=int,default=10)
    ap.add_argument("--initial",type=int,default=20);ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    print(run_loop(pd.read_csv(a.data),a.out,a.mode,a.reviews,a.rounds,a.budget,a.seed,a.llm_jsonl,a.initial).to_string(index=False))
if __name__=="__main__":cli()

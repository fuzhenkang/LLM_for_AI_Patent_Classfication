"""可信度加权软标签蒸馏: 离线teacher CSV -> TF-IDF Logistic Regression学生。"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score,roc_auc_score
from sklearn.model_selection import train_test_split

def fit_student(X,teacher_probs,confidence=None,hard_ids=None,hard_labels=None,
                alpha=.7,seed=42):
    """在同一行正负复制、以软标签为权重拟合交叉熵; alpha为教师权重。"""
    if not 0<=alpha<=1: raise ValueError("alpha must be between 0 and 1")
    t=np.asarray(teacher_probs,dtype=float)
    if len(t)!=X.shape[0] or not np.isfinite(t).all() or ((t<0)|(t>1)).any():
        raise ValueError("Invalid teacher probabilities")
    w=np.ones(len(t)) if confidence is None else np.asarray(confidence,dtype=float)
    if len(w)!=len(t) or np.any(w<0): raise ValueError("Invalid confidence")
    targets=t.copy()
    if hard_ids is not None:
        hard_ids=np.asarray(hard_ids,dtype=int)
        labels=np.asarray(hard_labels,dtype=int)
        if len(hard_ids)!=len(labels) or set(np.unique(labels))-({0,1}):
            raise ValueError("Invalid hard labels")
        targets[hard_ids]=(1-alpha)*labels+alpha*t[hard_ids]
        w[hard_ids]=np.maximum(w[hard_ids],1.0)
    combined=np.clip(targets,0,1)
    student=LogisticRegression(max_iter=800,random_state=seed)
    from scipy.sparse import vstack,issparse
    XX=vstack([X,X]) if issparse(X) else np.vstack([X,X])
    yy=np.r_[np.ones(len(t)),np.zeros(len(t))]
    weights=np.r_[w*combined,w*(1-combined)]
    student.fit(XX,yy,sample_weight=weights)
    return student

def distillation_comparison(df,teacher_csv,out,seed=42):
    """严格test隔离；teacher必须外部交叉拟合，脚本不为teacher背书。"""
    required={"patent_id","label"}
    if not required.issubset(df):raise ValueError("Data must have patent_id and label")
    tdf=pd.read_csv(teacher_csv)
    if not {"patent_id","teacher_probability"}.issubset(tdf):raise ValueError("Teacher requires patent_id, teacher_probability")
    if df.patent_id.duplicated().any() or tdf.patent_id.duplicated().any():raise ValueError("Duplicate IDs")
    df=df.merge(tdf,on="patent_id",how="left",validate="one_to_one")
    if df.teacher_probability.isna().any():raise ValueError("Teacher probabilities missing")
    for c in ("title","abstract","claims"):
        if c not in df:df[c]=""
    y=df.label.to_numpy(dtype=float)
    known=np.where(np.isfinite(y))[0]
    if len(known)<30:raise ValueError("At least 30 expert labels needed")
    if set(np.unique(y[known]))!={0.,1.}:raise ValueError("Two classes required")
    fit_ids,test=train_test_split(known,test_size=.25,stratify=y[known],random_state=seed)
    anchor,dev=train_test_split(fit_ids,test_size=.25,stratify=y[fit_ids],random_state=seed)
    train=np.array(sorted(set(range(len(df)))-set(test)-set(dev)))
    text=df[["title","abstract","claims"]].fillna("").astype(str).agg(" ".join,axis=1)
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=20000)
    Xtr=vec.fit_transform(text.iloc[train]);Xt=vec.transform(text.iloc[test])
    local={int(k):i for i,k in enumerate(train)}
    gold_ids=np.array([local[k] for k in anchor])
    conf=df["teacher_confidence"].fillna(1).to_numpy()[train] if "teacher_confidence" in df else None
    variants={"supervised":None,"soft_distill":.999,"anchored_distill":.7,"anchored_low_teacher":.3}
    rows=[];pred=[]
    for name,alpha in variants.items():
        if alpha is None:
            m=LogisticRegression(max_iter=800,random_state=seed).fit(vec.transform(text.iloc[anchor]),y[anchor].astype(int))
        else:
            m=fit_student(Xtr,df.teacher_probability.to_numpy()[train],conf,
                          gold_ids,y[anchor].astype(int),alpha,seed)
        p=m.predict_proba(Xt)[:,1]; predy=(p>=.5).astype(int);truth=y[test].astype(int)
        rows.append({"variant":name,"precision":precision_score(truth,predy,zero_division=0),
                     "recall":recall_score(truth,predy,zero_division=0),
                     "f1":f1_score(truth,predy,zero_division=0),
                     "pr_auc":average_precision_score(truth,p),"auroc":roc_auc_score(truth,p),
                     "student_train_samples":len(anchor) if alpha is None else len(train)})
        pred.extend({"variant":name,"patent_id":df.iloc[i].patent_id,"gold":int(truth[j]),"probability":float(p[j])} for j,i in enumerate(test))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"distillation_metrics.csv",index=False)
    pd.DataFrame(pred).to_csv(out/"distillation_predictions.csv",index=False)
    (out/"distillation_splits.json").write_text(json.dumps({"anchor":df.iloc[anchor].patent_id.tolist(),
      "dev":df.iloc[dev].patent_id.tolist(),"test":df.iloc[test].patent_id.tolist()},ensure_ascii=False,indent=2),encoding="utf-8")
    return rows

if __name__=="__main__":
    p=argparse.ArgumentParser(description="KE-WSPI 离线概率蒸馏消融")
    p.add_argument("--data",required=True);p.add_argument("--teacher",required=True)
    p.add_argument("--out",required=True);p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    print(json.dumps(distillation_comparison(pd.read_csv(a.data),a.teacher,a.out,a.seed),ensure_ascii=False,indent=2))

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from active_learning import score_candidates,select_batch,offline_learning_curve
from distillation import fit_student,distillation_comparison

def test_select_only_eligible_and_budget():
    p=np.array([.5,.9,.1,.4,.8])
    ids=select_batch(p,2,np.array([0,2,4]),strategy="entropy")
    assert len(ids)==2 and set(ids).issubset({0,2,4})
    assert ids[0]==0

def test_disagreement_and_abstain():
    v=np.array([[1,-1,0],[1,1,0],[0,0,0]])
    out=score_candidates([.5,.5,.5],votes=v,strategy="disagreement")
    assert out[0]>out[1] and out[2]==0

def test_soft_student():
    X=csr_matrix([[1.,0],[0,1.],[1,0],[0,1],[1,0],[0,1]])
    m=fit_student(X,[.95,.05,.95,.05,.95,.05],hard_ids=[0,1],hard_labels=[1,0])
    assert m.predict_proba(X)[0,1]>m.predict_proba(X)[1,1]

def test_offline_and_distillation(tmp_path):
    rows=[]
    for i in range(120):
        label=i%3==0
        rows.append(dict(patent_id=f"p{i}",label=int(label),title="Example",
                         abstract="neural network detection" if label else "power transformer coil",
                         claims="",cpc="G06N" if label else "H01F"))
    df=pd.DataFrame(rows)
    curve,queues=offline_learning_curve(df,initial=12,batch=5,rounds=2)
    assert len(curve)==12
    assert set(curve["strategy"])=={"random","entropy","disagreement","hybrid"}
    teacher=df[["patent_id"]].copy()
    teacher["teacher_probability"]=np.where(df.label==1,.9,.1)
    path=tmp_path/"teacher.csv";teacher.to_csv(path,index=False)
    metrics=distillation_comparison(df,str(path),str(tmp_path/"out"))
    assert len(metrics)==4
    assert (tmp_path/"out"/"distillation_metrics.csv").exists()

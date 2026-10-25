import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kewspi import weak_functions,evidence_lf,ECLR,collapse

def test_abstain():
    import pandas as pd
    df=pd.DataFrame([{"patent_id":"1","title":"","abstract":"unrelated", "claims":"","cpc":"A61B"}])
    assert weak_functions(df).tolist()==[[0,0,0]]

def test_collapse():
    assert collapse(np.array([[1,1,-1,0],[1,-1,0,0]])).tolist()==[[1,-1,0],[0,0,0]]

def test_anchor_model():
    L=np.array([[1,1],[-1,-1],[1,0],[-1,0]]*12)
    anchors=np.array([0,1,4,5,8,9,12,13])
    model=ECLR().fit(L,anchors,np.array([1,0,1,0,1,0,1,0]))
    assert model.predict(L)[0]>model.predict(L)[1]

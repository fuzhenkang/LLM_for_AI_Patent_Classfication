import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest
from closed_loop import run_loop

def synthetic(n=150):
    return pd.DataFrame([
        {"patent_id":f"CL{i:04}","label":int(i%3==0),
         "title":"Patent",
         "abstract":"neural network deep learning anomaly prediction" if i%3==0 else "electrical power transformer coil",
         "claims":"neural network classifier" if i%3==0 else "voltage signal winding",
         "cpc":"G06N" if i%3==0 else "H01F"}
        for i in range(n)
    ])

def test_oracle_loop_does_not_reuse_test(tmp_path):
    df=synthetic()
    result=run_loop(df,tmp_path/"oracle",mode="oracle",rounds=2,budget=4,initial=12)
    assert len(result)==3
    assert result.verified_train.iloc[-1]>=result.verified_train.iloc[0]
    assert (tmp_path/"oracle"/"closed_loop_metrics.csv").exists()
    assert (tmp_path/"oracle"/"review_queue_round_1.csv").exists()
    import json
    splits=json.loads((tmp_path/"oracle"/"split_ids.json").read_text())
    test=set(splits["test"])
    for path in sorted((tmp_path/"oracle").glob("review_queue_round_*.csv")):
        assert not (set(pd.read_csv(path).patent_id)&test)

def test_manual_review_does_not_fake_feedback(tmp_path):
    df=synthetic()
    result=run_loop(df,tmp_path/"manual",mode="review",rounds=2,budget=4,initial=12)
    assert result.verified_train.iloc[-1]==12
    import json
    split=json.loads((tmp_path/"manual"/"split_ids.json").read_text())
    assert len(split["pending_review"])>0

def test_oracle_missing_labels_rejected(tmp_path):
    df=synthetic()
    df.loc[30:70,"label"]=float("nan")
    with pytest.raises((ValueError,Exception)):
        run_loop(df,tmp_path/"missing",mode="oracle",rounds=2,budget=15,initial=12)

import numpy as np,pandas as pd,torch
from label_graph import DependentLabelModel
from rag_llm import weak_labels,KnowledgeRetriever,quote_supported
from neural_distill import PatentDataset

def test_label_model_and_dependency():
 x=np.array([[1,1,0,0,0],[-1,0,-1,0,0],[1,0,0,1,0],[-1,0,-1,0,0]]*6)
 gold=np.array([0,1,4,5,8,9,12,13])
 y=np.array([1,0,1,0,1,0,1,0])
 model=DependentLabelModel(5,1,[(0,1)])
 model.fit(x,gold,y,epochs=8)
 p=model.predict(x)
 assert len(p)==len(x) and np.isfinite(p).all()
 assert (p>=0).all() and (p<=1).all()

def test_weak_rules():
 df=pd.DataFrame([dict(title="",abstract="neural network classifier",claims="",cpc="G06N3/08"),
   dict(title="",abstract="electrical transformer winding",claims="",cpc="H01F")])
 r=weak_labels(df)
 assert r[0,0]==1 and r[0,1]==1 and r[1,2]==-1
 assert quote_supported("neural network","This uses NEURAL NETWORK")
 assert not quote_supported("neural network","This uses a transformer")

def test_retrieval():
 import json,tempfile,pathlib
 with tempfile.TemporaryDirectory() as p:
  path=pathlib.Path(p)/"k.json"
  path.write_text(json.dumps([{"id":"x","title":"Neural network","definition":"Machine learning classification"}]))
  assert KnowledgeRetriever(path).search("neural network")[0]["id"]=="x"

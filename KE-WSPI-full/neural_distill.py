"""PatentBERT/SciBERT-compatible neural student distillation.
Local model cache or Hugging Face identifier. Teacher probabilities must be OOF for research.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F

class PatentDataset(Dataset):
 def __init__(self,texts,probabilities,weights,labels,tokenizer,max_length):
  self.texts=list(texts);self.p=np.asarray(probabilities,dtype=float)
  self.weights=np.asarray(weights,dtype=float);self.labels=np.asarray(labels,dtype=float)
  self.tokenizer=tokenizer;self.max_length=max_length
 def __len__(self):return len(self.texts)
 def __getitem__(self,i):
  z=self.tokenizer(self.texts[i],max_length=self.max_length,truncation=True,padding="max_length",return_tensors="pt")
  return {**{k:v.squeeze(0) for k,v in z.items()},
   "soft":torch.tensor(self.p[i],dtype=torch.float32),
   "weight":torch.tensor(self.weights[i],dtype=torch.float32),
   "hard":torch.tensor(self.labels[i] if np.isfinite(self.labels[i]) else -1,dtype=torch.float32)}

def train_student(df,soft_probs,config,output,expert_labels=None,weights=None,device=None):
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 model_name=config.get("model_name","allenai/scibert_scivocab_uncased")
 token=AutoTokenizer.from_pretrained(model_name)
 model=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2,
                              ignore_mismatched_sizes=True)
 device=device or ("cuda" if torch.cuda.is_available() else "cpu")
 model.to(device)
 texts=df[["title","abstract","claims"]].fillna("").astype(str).agg(" ".join,axis=1).tolist()
 n=len(texts)
 labels=np.full(n,np.nan) if expert_labels is None else np.asarray(expert_labels,dtype=float)
 w=np.ones(n) if weights is None else np.asarray(weights,dtype=float)
 data=PatentDataset(texts,soft_probs,w,labels,token,int(config.get("max_length",256)))
 loader=DataLoader(data,batch_size=int(config.get("batch_size",8)),shuffle=True)
 optimizer=torch.optim.AdamW(model.parameters(),lr=float(config.get("learning_rate",2e-5)))
 temp=float(config.get("temperature",2.0));temp=max(temp,1e-3)
 alpha=float(config.get("distill_weight",.7))
 history=[]
 for epoch in range(int(config.get("epochs",1))):
  model.train();s=0
  for batch in loader:
   batch={k:v.to(device) for k,v in batch.items()}
   labels=batch.pop("hard");weight=batch.pop("weight");soft=batch.pop("soft")
   logits=model(**batch).logits
   # Binary teacher probabilities softened at temperature T
   tt=torch.stack([1-soft,soft],dim=-1).clamp(1e-5,1-1e-5)
   tt=F.softmax(torch.log(tt)/temp,dim=-1)
   kl=F.kl_div(F.log_softmax(logits/temp,dim=-1),tt,reduction="none").sum(-1)*(temp**2)
   valid=labels>=0
   hard=torch.zeros_like(kl)
   if valid.any():hard[valid]=F.cross_entropy(logits[valid],labels[valid].long(),reduction="none")
   # on gold, combine hard and soft; on unlabeled use only soft
   total=torch.where(valid,(1-alpha)*hard+alpha*kl,kl)
   loss=(total*weight).sum()/weight.sum().clamp_min(1e-6)
   optimizer.zero_grad();loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   optimizer.step();s+=float(loss.detach())
  history.append({"epoch":epoch+1,"loss":s/max(1,len(loader))})
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 model.save_pretrained(out);token.save_pretrained(out)
 (out/"train_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
 return model,token,history

@torch.no_grad()
def predict_student(model,tokenizer,df,max_length=256,batch_size=16):
 device=next(model.parameters()).device
 text=df[["title","abstract","claims"]].fillna("").astype(str).agg(" ".join,axis=1).tolist()
 model.eval();pred=[]
 for i in range(0,len(text),batch_size):
  inp=tokenizer(text[i:i+batch_size],padding=True,truncation=True,max_length=max_length,return_tensors="pt")
  logits=model(**{k:v.to(device) for k,v in inp.items()}).logits
  pred.extend(logits.softmax(-1)[:,1].cpu().tolist())
 return np.asarray(pred)

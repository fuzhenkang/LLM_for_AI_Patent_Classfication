"""Bidirectional Spy reconstruction with explicitly reported departures from original human CPC filter."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score

def spy_refinement(texts,seed_positive,seed_negative,spy_neg=100000,spy_pos=50000,
                   target_neg=.85,target_pos=.99,random_state=42):
 """2-pass spy protocol. Counts capped by source pool size. Thresholds chosen from spy quantiles.
 Original article's manual CPC exclusion must be supplied externally, cannot infer exact decisions.
 Returns -1,0,+1 (negative,unselected,positive).
 """
 rng=np.random.default_rng(random_state)
 positive=np.asarray(seed_positive,dtype=int)
 negative=np.asarray(seed_negative,dtype=int)
 if len(positive)<3 or len(negative)<3:raise ValueError("Need >=3 distinct positive and negative seed patents")
 if np.intersect1d(positive,negative).size:raise ValueError("positive and negative pools overlap")
 x=TfidfVectorizer(max_features=15000,ngram_range=(1,2)).fit_transform(texts)
 # First: negative spies placed into presumed positive, classifier aims to recover negatives.
 spies=rng.choice(negative,size=min(spy_neg,max(1,len(negative)//5)),replace=False)
 pos_mixed=np.r_[positive,spies]
 neg_train=np.setdiff1d(negative,spies)
 clf=LogisticRegression(max_iter=800).fit(x[np.r_[pos_mixed,neg_train]],
               np.r_[np.ones(len(pos_mixed)),np.zeros(len(neg_train))])
 score=clf.predict_proba(x)[:,1]
 # 85% spies should be below the threshold.
 threshold1=np.quantile(score[spies],target_neg)
 pos_clean=positive[score[positive]>threshold1]
 if len(pos_clean)<3:pos_clean=positive[np.argsort(score[positive])[-max(3,len(positive)//2):]]
 # Second: positive spies placed in negatives, detect wrongly negative candidates.
 spies2=rng.choice(pos_clean,size=min(spy_pos,max(1,len(pos_clean)//5)),replace=False)
 neg_mixed=np.r_[negative,spies2]
 pos_train=np.setdiff1d(pos_clean,spies2)
 clf2=LogisticRegression(max_iter=800).fit(x[np.r_[pos_train,neg_mixed]],
            np.r_[np.ones(len(pos_train)),np.zeros(len(neg_mixed))])
 score2=clf2.predict_proba(x)[:,1]
 # 99% positive spies at or above threshold -> 1st percentile.
 threshold2=np.quantile(score2[spies2],1-target_pos)
 neg_clean=negative[score2[negative]<threshold2]
 lbl=np.zeros(len(texts),dtype=int);lbl[pos_clean]=1;lbl[neg_clean]=-1
 return lbl,{"spy_negative_count":len(spies),"spy_positive_count":len(spies2),
             "negative_spy_quantile":threshold1,"positive_spy_quantile":threshold2,
             "positive_kept":len(pos_clean),"negative_kept":len(neg_clean),
             "original_manual_CPC_filter":"NOT INCLUDED"}

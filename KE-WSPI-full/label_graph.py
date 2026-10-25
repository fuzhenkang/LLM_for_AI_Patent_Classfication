"""Sparse pairwise-dependent weak-source latent label model, semi-supervised MLE.
P(L|Y,z) proportional exp(unary source-class scores + sparse pair/class potentials).
Gold labels anchor class direction; class prior learned with Beta regularization.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class DependentLabelModel(nn.Module):
    def __init__(self,n_sources,n_groups=1,pairs=None):
        super().__init__()
        self.n_sources=n_sources;self.n_groups=n_groups
        self.pairs=pairs if pairs is not None else [(0,1),(1,3)]
        self.pairs=[(a,b) for a,b in self.pairs if a<n_sources and b<n_sources]
        self.unary=nn.Parameter(torch.zeros(n_groups,2,n_sources,3))
        self.pair=nn.Parameter(torch.zeros(len(self.pairs),2,3,3))
        self.prior_logits=nn.Parameter(torch.tensor(-1.0))
        self.register_buffer("grid",torch.cartesian_prod(*[torch.arange(3) for _ in range(n_sources)]))
        if n_sources>8:raise ValueError("Exact partition limited to <=8 sources; group rules before fitting")
    def energy(self,L,G):
        """unnormalized conditional log-likelihood for each possible class.
        L shape B,M in {0,1,2} for -1,abstain,+1.
        """
        b=len(L)
        u=self.unary[G] # B,2,M,3
        t=torch.zeros((b,2),device=L.device)
        for j in range(self.n_sources):
            t=t+u[:,:,j,:].gather(-1,L[:,j,None,None].expand(-1,2,1)).squeeze(-1)
        for k,(a,c) in enumerate(self.pairs):
            t=t+self.pair[k,:,L[:,a],L[:,c]].transpose(0,1)
        return t
    def log_partition(self,device):
        """group and class conditional normalization Z(y,g) via exhaustive 3^m labels."""
        allL=self.grid.to(device)
        z=[]
        for g in range(self.n_groups):
            GG=torch.full((len(allL),),g,device=device,dtype=torch.long)
            z.append(torch.logsumexp(self.energy(allL,GG),dim=0))
        return torch.stack(z)
    def log_joint(self,L,G):
        z=self.log_partition(L.device)
        e=self.energy(L,G)-z[G]
        pi=F.logsigmoid(self.prior_logits)
        return e+torch.stack([F.logsigmoid(-self.prior_logits),pi]).unsqueeze(0)
    @torch.no_grad()
    def predict(self,L,groups=None):
        self.eval()
        x=torch.as_tensor(np.asarray(L,dtype=np.int64)+1,dtype=torch.long)
        g=torch.as_tensor(np.zeros(len(x),dtype=np.int64) if groups is None else groups,dtype=torch.long)
        return self.log_joint(x,g).softmax(-1)[:,1].cpu().numpy()
    def fit(self,L,gold_idx,gold_labels,groups=None,epochs=120,lr=.025,pair_l2=.2,seed=42):
        torch.manual_seed(seed)
        raw=np.asarray(L,dtype=np.int64)
        if not np.isin(raw,[-1,0,1]).all():raise ValueError("Labels must be -1/0/1")
        idx=np.asarray(gold_idx,dtype=int);y=np.asarray(gold_labels,dtype=int)
        if not len(idx) or set(np.unique(y))!={0,1}:raise ValueError("Need expert examples from both classes")
        x=torch.tensor(raw+1,dtype=torch.long)
        g=torch.tensor(np.zeros(len(raw),dtype=np.int64) if groups is None else np.asarray(groups),dtype=torch.long)
        gg=torch.tensor(idx,dtype=torch.long);yy=torch.tensor(y,dtype=torch.long)
        opt=torch.optim.Adam(self.parameters(),lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            joint=self.log_joint(x,g)
            unsup=-torch.logsumexp(joint,dim=1).mean()
            # log likelihood of observed source outputs AND expert labels
            sup=-joint[gg,yy].mean()
            regular=.02*self.unary.square().mean()
            if len(self.pairs):regular=regular+pair_l2*self.pair.square().mean()
            loss=unsup+sup+regular
            loss.backward();opt.step()
        return self

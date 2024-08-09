import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCN
import scipy.sparse as sp
from utils import process
class GPPT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers=2,
                 activation=F.relu,
                 dropout=0.5,
                 center_num=7):
        super(GPPT, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.n_classes=n_classes
        self.center_num=center_num
        # input layer
        self.layers.append(GCN(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCN(n_hidden, n_hidden))

        self.prompt=nn.Linear(2*n_hidden,self.center_num,bias=False)
        
        self.pp = nn.ModuleList()
        for i in range(self.center_num):
            self.pp.append(nn.Linear(2*n_hidden,n_classes,bias=False))
        
    def forward(self, feature, adj):
        if self.dropout==False:
            h=feature
        else:
            h = self.dropout(feature)
        for l, layer in enumerate(self.layers):
            # print("l",l)
            h_dst = h[:h.shape[0]]  # <---
            h = layer(h,adj)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout!=False:
                    h = self.dropout(h)
        h = self.activation(h)
        h_dst = self.activation(h_dst)
        neighbor=h_dst
        h=torch.cat((h,neighbor),dim=1)

        out=self.prompt(h)
        index=torch.argmax(out, dim=1)
        out=torch.FloatTensor(h.shape[0],self.n_classes).cuda()
        for i in range(self.center_num):
            out[index==i]=self.pp[i](h[index==i])
        return out
    
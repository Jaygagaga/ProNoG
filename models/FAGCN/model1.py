import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import numpy as np

from promptvector import PromptVector


class FALayer(nn.Module):
    def __init__(self,in_dim, dropout):
        super(FALayer, self).__init__()
        # self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h,g):
        g.ndata['h'] = h
        g.apply_edges(self.edge_applying)
        g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return g.ndata['z']


class FAGCN(nn.Module):
    def __init__(self,in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):

        super(FAGCN, self).__init__()
        # self.selfprompt = downstreamprompt(hidden_dim)
        # self.neighborsprompt = downstreamprompt(hidden_dim)
        # self.neighbors_2hopprompt = downstreamprompt(hidden_dim)
        # self.sample = torch.tensor(sample, dtype=int).cuda()
        # self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer( hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward_pretrain(self, h,g):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h,g)
            h = self.eps * raw + h
        # h = self.t2(h)
        # lploss = compareloss(h, self.sample, temperature=10)
        return h #F.log_softmax(h, 1)
    def forward(self, h,g):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        # raw = self.selfprompt(h)
        raw = h
        for i in range(self.layer_num):
            # h = self.selfprompt(h)
            h = self.layers[i](h,g)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)
    # def embed(self, h):
    #     h = F.dropout(h, p=self.dropout, training=self.training)
    #     h = torch.relu(self.t1(h))
    #     h = F.dropout(h, p=self.dropout, training=self.training)
    #     raw = h
    #     for i in range(self.layer_num):
    #         h = self.layers[i](h)
    #         h = self.eps * raw + h
    #     return h




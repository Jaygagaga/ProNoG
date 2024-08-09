import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import numpy as np

from promptvector import PromptVector
import torch_scatter
def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):

    cnt = 0

    result = torch.FloatTensor(graph_sizes.shape[0], batched_graph_feats.shape[1]).cuda()

    for i in range(graph_sizes.shape[0]):
        # print("i",i)
        current_graphlen = int(graph_sizes[i].item())
        graphlen = range(cnt,cnt+current_graphlen)
        # print("graphlen",graphlen)
        result[i] = torch.sum(batched_graph_feats[graphlen,:], dim=0)
        cnt = cnt + current_graphlen
    # print("resultsum",cnt)
    return result
# class FALayer(nn.Module):
#     def __init__(self,in_dim, dropout):
#         super(FALayer, self).__init__()
#         # self.g = g
#         self.dropout = nn.Dropout(dropout)
#         self.gate = nn.Linear(2 * in_dim, 1)
#         nn.init.xavier_normal_(self.gate.weight, gain=1.414)
#
#     def edge_applying(self, edges):
#         h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
#         g = torch.tanh(self.gate(h2)).squeeze()
#         e = g * edges.dst['d'] * edges.src['d']
#         e = self.dropout(e)
#         return {'e': e, 'm': g}
#
#     def forward(self, h,g):
#         g.ndata['h'] = h
#         g.apply_edges(self.edge_applying)
#         g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))
#
#         return g.ndata['z']
#
#
# class FAGCN(nn.Module):
#     def __init__(self,in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
#
#         super(FAGCN, self).__init__()
#         # self.selfprompt = downstreamprompt(hidden_dim)
#         # self.neighborsprompt = downstreamprompt(hidden_dim)
#         # self.neighbors_2hopprompt = downstreamprompt(hidden_dim)
#         # self.sample = torch.tensor(sample, dtype=int).cuda()
#         # self.g = g
#         self.eps = eps
#         self.layer_num = layer_num
#         self.dropout = dropout
#
#         self.layers = nn.ModuleList()
#         for i in range(self.layer_num):
#             self.layers.append(FALayer(hidden_dim, dropout))
#
#         self.t1 = nn.Linear(in_dim, hidden_dim)
#         self.t2 = nn.Linear(hidden_dim, out_dim)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_normal_(self.t1.weight, gain=1.414)
#         nn.init.xavier_normal_(self.t2.weight, gain=1.414)
#
#     def forward_pretrain(self, h,g):
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = torch.relu(self.t1(h))
#         h = F.dropout(h, p=self.dropout, training=self.training)
#
#         raw = h
#         for i in range(self.layer_num):
#             h = self.layers[i](h,g)
#             h = self.eps * raw + h
#         # h = self.t2(h)
#         # lploss = compareloss(h, self.sample, temperature=10)
#         return h #F.log_softmax(h, 1)
#     def forward(self, h):
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = torch.relu(self.t1(h))
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         # raw = self.selfprompt(h)
#         raw = h
#         for i in range(self.layer_num):
#             # h = self.selfprompt(h)
#             h = self.layers[i](h)
#             h = self.eps * raw + h
#         h = self.t2(h)
#         return F.log_softmax(h, 1)
class FALayer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(FALayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1) #1403*2 + 1403*256 + 256*5
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


class FAGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):

        super(FAGCN, self).__init__()
        # self.selfprompt = downstreamprompt(hidden_dim)
        # self.neighborsprompt = downstreamprompt(hidden_dim)
        # self.neighbors_2hopprompt = downstreamprompt(hidden_dim)
        # self.sample = torch.tensor(sample, dtype=int).cuda()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward_pretrain(self, h,g=None):
        if g!=None:
            for i in range(self.layer_num):
                self.layers.append(FALayer(g, self.hidden_dim, self.dropout))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        # h = self.t2(h)
        # lploss = compareloss(h, self.sample, temperature=10)
        return h #F.log_softmax(h, 1)
    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        # raw = self.selfprompt(h)
        raw = h
        for i in range(self.layer_num):
            # h = self.selfprompt(h)
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)

    def forward_graph(self, h, train_graph_nodes,idx):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        # raw = self.selfprompt(h)
        raw = h
        for i in range(self.layer_num):
            # h = self.selfprompt(h)
            h = self.layers[i](h)
            h = self.eps * raw + h
        output = h[train_graph_nodes]
        logits_graph = torch_scatter.scatter(src=output, index=torch.tensor(sum(idx,[])).cuda(), dim=0, reduce="sum")
        logits_graph = self.t2(logits_graph)
        logits_graph = F.log_softmax(logits_graph, 1)
        return logits_graph
    def forward1(self, h,train_graph_nodes=None, graph_len=None):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        # raw = self.selfprompt(h)
        raw = h
        for i in range(self.layer_num):
            # h = self.selfprompt(h)
            h = self.layers[i](h)
            h = self.eps * raw + h
        if train_graph_nodes != None:
            logits_ = []
            for i in train_graph_nodes:
                logits_.append(torch.sum(h.unsqueeze(0)[0, i], dim=0))
            logits_graph = torch.stack(logits_, dim=0)
        if graph_len !=None:
            logits_graph = split_and_batchify_graph_feats(h, graph_len)
        logits_graph = self.t2(logits_graph)
        logits_graph = F.log_softmax(logits_graph, 1)
        return logits_graph
    def embed(self, h,g,train_graph_nodes=None,graph_len=None):
        self.g = g
        for i in self.layers:
            i.g = g
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        # raw = self.selfprompt(h)
        raw = h
        for i in range(self.layer_num):
            # h = self.selfprompt(h)
            h = self.layers[i](h)
            h = self.eps * raw + h
        if train_graph_nodes != None:
            logits_ = []
            for i in train_graph_nodes:
                logits_.append(torch.sum(h.unsqueeze(0)[0, i], dim=0))
            logits_graph = torch.stack(logits_, dim=0)
        elif graph_len != None:
            logits_graph = split_and_batchify_graph_feats(h, graph_len)
        else:
            logits_graph=h
        logits_graph = self.t2(logits_graph)
        logits_graph = F.log_softmax(logits_graph, 1)
        return logits_graph



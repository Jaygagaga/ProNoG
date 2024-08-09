import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import copy
from functools import partial
from dgl.nn.pytorch.conv import RelGraphConv
# from basemodel import GraphAdjModel
import numpy as np
import tqdm
from models.gcnlayers import GcnLayers

class Lp(nn.Module):
    def __init__(self, n_in, n_h,):
        super(Lp, self).__init__()
        self.sigm = nn.ELU()
        self.act=torch.nn.LeakyReLU()
        # self.dropout=torch.nn.Dropout(p=config["dropout"])
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()



    def forward(self,gcn,seq,adj,LP=False):
        h_1 = gcn(seq,adj,LP=LP,)

        

        # 
        # ret = h_1 * self.prompt
        ret = h_1 
        # print("ret1",ret)
        ret = self.sigm(ret.squeeze(dim=0))
                # print("ret2",ret)
        # ret = ret.squeeze(dim=0)
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

def readout(h,adj,node):
    tempneighbors,tempneighbors_2hop = find_2hop_neighbors(adj,node)
    tempneighborsemb = torch.mean(h[0,tempneighbors],dim=0)
    tempneighbors_2hopsemb = torch.mean(h[0,tempneighbors_2hop],dim=0)
    selfemb = h[0,node]
    ret = torch.cat((selfemb,tempneighborsemb,tempneighbors_2hopsemb),0)
    return ret 


def find_neighbors(adj, node):
    neighbors = []
    for i in range(len(adj[node])):
        if adj[node][i] != 0:
            neighbors.append(i)
    return neighbors



#寻找当前节点的邻居节点和二阶邻居节点
def find_2hop_neighbors(adj, node):
    neighbors = []
    # print(type(adj))
    for i in range(len(adj[node])):
        # print('i',i)
        # print('node',node)
        # print('adj[node][i]',adj[node,i])
        if adj[node][i] != 0:
            neighbors.append(i)
    neighbors_2hop = []
    for i in neighbors:
        for j in range(len(adj[i])):
            if adj[i][j] != 0:
                neighbors_2hop.append(j)
    return neighbors, neighbors_2hop
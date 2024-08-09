import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LP import Lp
from models.gcnlayers import GcnLayers
from models.gppt import GPPT
from layers import GCN, AvgReadout
from torch_geometric.nn import GIN, GAT
from models.dgi_ import DGI
from models.H2GCN.model import H2GCN
from models.FAGCN.model import FAGCN
import tqdm
import numpy as np
import torch_scatter
import dgl
# from models.graphcl import GraphCL
class PrePrompt(nn.Module):
    def __init__(self, n_in, hidden_size,num_layers_num,p,neighborslist=None,neighbors_2hoplist=None,neighborsindex=None,neighbors_2hopindex=None,model_name='GIN', reduction='mean',
                 hop_level=2, gp=1,weight=0, concat_dense=0,type_vocab_size=2,pretrain_hop=0,
                 g=None,eps=0,aug_type="edge",sample_num=None,inputs_indices=None):
        super(PrePrompt, self).__init__()
        self.weight=weight
        self.reduction = reduction
        self.aug_type=aug_type
        self.lp = Lp(n_in, hidden_size)
        self.concat_dense=concat_dense

        # if concat_dense ==1:
        #     self.linear = nn.Linear(hidden_size*3, hidden_size)
        self.model_name = model_name
        self.pretrain_hop = pretrain_hop
        if self.pretrain_hop ==1:
            self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size).cuda()
            # self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.hop_level = hop_level
        if self.model_name == 'GAT':
            self.model = GAT(n_in, hidden_size, num_layers=num_layers_num)
        if self.model_name == 'GIN':
            self.model = GIN(n_in, hidden_size, num_layers=num_layers_num,
                             dropout=p, jk='cat')
        if self.model_name=='GCN':
            self.model = GcnLayers(n_in, hidden_size, num_layers_num, p)

        if self.model_name == 'FAGCN':
            self.model = FAGCN(g, n_in, hidden_size,hidden_size, p,eps, num_layers_num).cuda()
        if self.model_name == 'H2GCN':
            self.model = H2GCN(n_in,hidden_dim=hidden_size,class_dim=hidden_size,use_relu=False).cuda()
        if self.model_name == 'DGI':
            self.model  = DGI(g,n_in,hidden_size,num_layers_num,nn.PReLU(hidden_size),p)
        #g, in_dim, hidden_dim, out_dim, dropout, eps,
        # if self.model_name == 'GraphCL':
        #     self.model_ = FAGCN(n_in, hidden_size, hidden_size, p, eps, num_layers_num).cuda() #in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2)
        #     self.model = GraphCL(hidden_size).cuda() #gcn, feature1, feature2, feature3, feature4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2,
        self.read = AvgReadout()
        # self.neighborslist = neighborslist
        # self.neighbors_2hoplist = neighbors_2hoplist
        # if neighborsindex != None:
        #     self.neighborsindex = torch.LongTensor(neighborsindex)
        #     self.neighbors_2hopindex = torch.LongTensor(neighbors_2hopindex)
        self.gp = gp
        # self.logits_= nn.Parameter(torch.FloatTensor(nb_nodes,hidden_size), requires_grad=True)


        # self.sample =sample
        # self.inputs_adj = inputs_adj.cuda()
        self.inputs_indices = inputs_indices
        self.sample_num = sample_num

        self.g = g
        # print("sample",self.sample)
        if self.weight==1:
            self.tokens = weighted_feature(self.hop_level+1)
        # self.dffprompt = weighted_prompt(2)
        self.loss = nn.BCEWithLogitsLoss()
        self.act = nn.ELU()

    def readout(self,inputs, neighborslist, neighbors_2hoplist, neighborsindex, neighbors_2hopindex):
        src1 = torch.mm(neighborslist, inputs)
        # print(neighborsindex.shape)
        # print(src1.shape)
        src2 = torch.mm(neighbors_2hoplist, inputs)
        if self.reduction == 'mean':
            neighborsemb = torch_scatter.scatter(src=src1, index=neighborsindex, dim=0, reduce="mean")
            neighbors_2hopsemb = torch_scatter.scatter(src=src2, index=neighbors_2hopindex, dim=0, reduce="mean")
        else:
            neighborsemb = torch_scatter.scatter(src=src1, index=neighborsindex, dim=0, reduce="sum")
            neighbors_2hopsemb = torch_scatter.scatter(src=src2, index=neighbors_2hopindex, dim=0, reduce="sum")
        return neighborsemb, neighbors_2hopsemb

    def forward(self, feature=None, edge_index=None,adj=None,temperature=10,LP=False):
        if self.model_name != "DGI":

            if self.model_name == 'GAT' or self.model_name == 'GIN':
                logits_ = self.model(feature, edge_index)
            elif self.model_name == 'FAGCN':
                self.logits_ = self.model.forward_pretrain(feature)
            elif self.model_name == 'H2GCN':
                self.logits_ = self.model.forward_pretrain(adj, feature)
            else:
                logits_ = self.model(feature, adj,LP=LP)
                self.logits_ = logits_.squeeze(0)
    
            if self.reduction == 'sum':
                subgraph_embeddings = torch_scatter.scatter(src=self.logits_, index=self.inputs_indices, dim=0, reduce="sum")
            else:
                subgraph_embeddings = torch_scatter.scatter(src=self.logits_, index=self.inputs_indices, dim=0, reduce="mean")
            self.self_index =  [num*(self.sample_num+2)  for num in range(0,int(subgraph_embeddings.size(0)/(self.sample_num+2)))]
    
            self_logits = torch.index_select(subgraph_embeddings,0,torch.tensor(self.self_index).cuda()).unsqueeze(1).repeat(1,self.sample_num+1,1)
            subgraph_logits = torch.index_select(subgraph_embeddings, 0, torch.tensor([n for n in range(0,subgraph_embeddings.size(0)) if n not in self.self_index]).cuda()).reshape(self_logits.size(0),-1,self_logits.size(-1))
    
            sim = F.cosine_similarity(self_logits, subgraph_logits, dim=2)
            # print("sim",sim)
            exp = torch.exp(sim)
            exp = exp / temperature
            exp = exp.permute(1, 0)
            numerator = exp[0].reshape(-1, 1)
            denominator = exp[1:exp.size(0)]
            denominator = denominator.permute(1, 0)
            denominator = denominator.sum(dim=1, keepdim=True)
    
            # print("numerator",numerator)
            # print("denominator",denominator)
            res = -1 * torch.log(numerator / denominator)
            loss = res.mean()
    

        else:
            loss = self.model(feature)
        return loss



    def embed(self, feature=None, adj=None, msk=None,LP=False,edge_index=None,g=None):
        # print("feature",feature.shape)
        # print("adj",adj.shape)
        if self.model_name == 'GAT' or self.model_name=='GIN':
            feature = torch.squeeze(feature, 0)
            h_1 = self.model(feature, edge_index)
        elif self.model_name == 'FAGCN':
            self.model.g = g
            for i in self.model.layers:
                i.g = g

            h_1 = self.model.forward_pretrain(feature)
        elif self.model_name == 'H2GCN':
            h_1 = self.model.embed(adj,feature)
        elif self.model_name == 'DGI':
            h_1 = self.model.forward_embed(feature)
        else:

            h_1 = self.model(feature, adj,LP=LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()
    


class weighted_feature(nn.Module):
    def __init__(self,weightednum):
        super(weighted_feature, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(1)
        # self.weight[0][1].data.fill_(1)
        # if len(self.weight[0]) ==3:
        #     self.weight[0][-1].data.fill_(10)
    def forward(self, graph_embedding1,graph_embedding2=None,graph_embedding3=None):
        # print("weight",self.weight)
        if graph_embedding2 != None:
            graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2 + self.weight[0][-1] * graph_embedding3
        else:
            graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][-1] * graph_embedding3
        return self.act(graph_embedding)


def mygather(feature, index):
    # print("index",index)
    # print("indexsize",index.shape)  
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    # print(tuples)

    # print("feature",feature)
    # print("featuresize",feature.shape)
    # print("index",index)
    # print("indexsize",index.shape)
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))


# def compareloss(logits,sample,temperature,sample_adjs,sample_idx):
#
#
#     h_tuples=mygather(logits,sample)
#     # print("tuples",h_tuples)
#     temp = torch.arange(0, len(sample))
#     temp = temp.reshape(-1, 1)
#     temp = torch.broadcast_to(temp, (temp.size(0), sample.size(1)))
#     # temp = m(temp)
#     temp=temp.cuda()
#     h_i = mygather(logits, temp)
#     # print("h_i",h_i)
#     # print("h_tuple",h_tuples)
#     sim = F.cosine_similarity(h_i, h_tuples, dim=2)
#     # print("sim",sim)
#     exp = torch.exp(sim)
#     exp = exp / temperature
#     exp = exp.permute(1, 0)
#     numerator = exp[0].reshape(-1, 1)
#     denominator = exp[1:exp.size(0)]
#     denominator = denominator.permute(1, 0)
#     denominator = denominator.sum(dim=1, keepdim=True)
#
#     # print("numerator",numerator)
#     # print("denominator",denominator)
#     res = -1 * torch.log(numerator / denominator)
#     return res.mean()


def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    print("#############")
    print("start sampling disconnected tuples")
    for i in tqdm.trange(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)

    




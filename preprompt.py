import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gcnlayers import GcnLayers
from models.gppt import GPPT
from layers import GCN, AvgReadout
from torch_geometric.nn import GIN, GAT
from models.H2GCN.model import H2GCN
from models.FAGCN.model import FAGCN
import tqdm
import numpy as np
import torch_scatter
class PrePrompt(nn.Module):
    def __init__(self, n_in, hidden_size,num_layers_num,p,sample=None,neighborslist=None,neighbors_2hoplist=None,neighborsindex=None,neighbors_2hopindex=None,
                 model_name='GCN', reduction='mean',
                 hop_level=2, gp=1,weight=0, concat_dense=0,type_vocab_size=2,pretrain_hop=0,
                 g=None,eps=0):
        super(PrePrompt, self).__init__()
        self.weight=weight
        self.reduction = reduction
        # self.lp = Lp(n_in, hidden_size)
        self.concat_dense=concat_dense
        self.sigm = nn.ELU()

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
            self.model = FAGCN(g, n_in, hidden_size,hidden_size, p,eps, num_layers_num).cuda() #g, in_dim, hidden_dim, out_dim, dropout, eps,
        self.read = AvgReadout()
        if neighborslist !=None:
            self.neighborslist = neighborslist.cuda()
            self.neighbors_2hoplist = neighbors_2hoplist.cuda()
            self.neighborsindex = torch.LongTensor(neighborsindex).cuda()
            self.neighbors_2hopindex = torch.LongTensor(neighbors_2hopindex).cuda()
        self.gp = gp

        self.sample = sample
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


    def forward(self, feature, adj=None,g=None,LP=False,edge_index=None,sample=None,neighborslist=None, neighbors_2hoplist=None, neighborsindex=None, neighbors_2hopindex=None):

        if self.model_name == 'GAT' or self.model_name == 'GIN':
            logits3 = self.model(feature, edge_index)
        elif self.model_name == 'FAGCN':
            if g != None:
                for i in self.model.layers:
                    i.g = g
                logits3 = self.model.forward_pretrain(feature,g)
            else:
                logits3 = self.model.forward_pretrain(feature)

        else:
            if feature.dim() == 2:
                feature = torch.squeeze(feature, 0)
            logits3 = self.model(feature,adj,LP=LP)
            logits3 = self.sigm(logits3.squeeze(dim=0))
        # token_type_ids_center = torch.zeros((logits3.shape[0]), dtype=torch.long, device=logits3.device)
        # if self.pretrain_hop==1:
        #     logits3_ = logits3 + self.token_type_embeddings(token_type_ids_center)
        #     center_embedding = self.linear1(logits3_)
        # else:
        #     center_embedding = logits3
        center_embedding = logits3
        if self.gp ==0: #no subgraph
            logits=logits3
            # logits=logits.squeeze(0)


        else: # Train weights, 2hop/1hop
            if neighbors_2hoplist != None:
                neighboremb, neighbors_2hopemb = self.readout(logits3, neighborslist, neighbors_2hoplist,
                                                              neighborsindex, neighbors_2hopindex)
            else:

                neighboremb,neighbors_2hopemb = self.readout(logits3,self.neighborslist,self.neighbors_2hoplist,self.neighborsindex,self.neighbors_2hopindex)

            #
            # if self.pretrain_hop==1:
            #     token_type_ids_1hop = torch.zeros((neighboremb.shape[0]), dtype=torch.long,
            #                                       device=neighboremb.device) + 0
            #     token_type_ids_2hop = torch.zeros((neighbors_2hopemb.shape[0]), dtype=torch.long,
            #                                       device=neighbors_2hopemb.device) + 1
            #     neighborsembds_ = neighboremb + self.token_type_embeddings(token_type_ids_1hop)
            #     neighbors_embedding =self.linear2(neighborsembds_)
            #     neighbors_2hopembds_ = neighbors_2hopemb + self.token_type_embeddings(token_type_ids_2hop)
            #     neighbors2hop_embedding = self.linear3(neighbors_2hopembds_)
            # else:
            neighbors_embedding= neighboremb
            neighbors2hop_embedding = neighbors_2hopemb

            if self.weight==1: #train weights
                if self.hop_level == 1:
                    neighbors_2hopemb = None
                # logits = self.tokens(neighboremb,neighbors_2hopemb,logits3)
                logits = self.tokens(neighbors_embedding, neighbors2hop_embedding, center_embedding)
            else: #no not train weights
                if self.hop_level == 2:
                    # neighbors2hop_embedding = None
                    stacked = torch.stack((neighbors_embedding,neighbors2hop_embedding,center_embedding),0)
                    # stacked = torch.stack((neighborsembds_, neighbors_2hopembds_, logits3_), 0)
                    logits = torch.sum(stacked,0)
                else:
                    stacked = torch.stack((neighbors_embedding, center_embedding), 0)
                    logits = torch.sum(stacked, 0)

        if sample !=None:
            lploss = compareloss(logits, sample, temperature=10)
        else:
            lploss = compareloss(logits, self.sample, temperature=10)

        ret = lploss
        return ret

    def embed(self, feature, adj=None, msk=None,LP=False,edge_index=None, g=None):
        # print("feature",feature.shape)
        # print("adj",adj.shape)
        if self.model_name == 'GAT' or self.model_name=='GIN':
            feature = torch.squeeze(feature, 0)
            h_1 = self.model(feature, edge_index)
        elif self.model_name == 'FAGCN':
            if g != None:
                for i in self.model.layers:
                    i.g = g
            h_1 = self.model.forward_pretrain(feature)
        else:
            h_1 = self.model(feature, adj,LP=LP,GPextension=False)
            h_1 = self.sigm(h_1.squeeze(dim=0))

        # c = self.read(h_1, msk)

        return h_1.detach(), None
    


# class weighted_feature(nn.Module):
#     def __init__(self,weightednum):
#         super(weighted_feature, self).__init__()
#         self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
#         self.act = nn.ELU()
#         self.reset_parameters()
#     def reset_parameters(self):
#         # torch.nn.init.xavier_uniform_(self.weight)
#
#         self.weight[0][0].data.fill_(1)
#         self.weight[0][1].data.fill_(1)
#         if len(self.weight[0]) ==3:
#             self.weight[0][-1].data.fill_(10)
#     def forward(self, graph_embedding1,graph_embedding2=None,graph_embedding3=None):
#         # print("weight",self.weight)
#         if graph_embedding2 != None:
#             graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2 + self.weight[0][-1] * graph_embedding3
#         else:
#             graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][-1] * graph_embedding3
#         return self.act(graph_embedding)

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


def compareloss(feature,tuples,temperature):

    h_tuples=mygather(feature,tuples)
    # print("tuples",h_tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    # temp = m(temp)
    temp=temp.cuda()
    h_i = mygather(feature, temp)
    # print("h_i",h_i)
    # print("h_tuple",h_tuples)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
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
    return res.mean()
def prompt_pretrain_sample(adj,n):
    #print("adj.shape", adj.shape)
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    # print("#############")
    # print("start sampling disconnected tuples")
    for i in range(nodenum):
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

# def prompt_pretrain_sample(adj,n):
#     nodenum=adj.shape[0]
#     indices=adj.indices
#     indptr=adj.indptr
#     res=np.zeros((nodenum,1+n))
#     whole=np.array(range(nodenum))
#     print("#############")
#     print("start sampling disconnected tuples")
#     for i in tqdm.trange(nodenum):
#         nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
#         zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
#         np.random.shuffle(nonzero_index_i_row)
#         np.random.shuffle(zero_index_i_row)
#         if np.size(nonzero_index_i_row)==0:
#             res[i][0] = i
#         else:
#             res[i][0]=nonzero_index_i_row[0]
#         res[i][1:1+n]=zero_index_i_row[0:n]
#     return res.astype(int)

    




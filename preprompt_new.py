import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gcnlayers import GcnLayers
from models.gppt import GPPT
# from layers import GCN, AvgReadout
from torch_geometric.nn import GIN, GAT
from models.H2GCN.model import H2GCN
from models.FAGCN.model import FAGCN
import tqdm
import numpy as np
import torch_scatter
import dgl
import random
from graphcl.gcn import GCN
class PrePrompt(nn.Module):
    def __init__(self, n_in, hidden_size,num_layers_num,p,model_name='GIN', reduction='mean',
                 hop_level=2, gp=1,weight=0, concat_dense=0,type_vocab_size=2,pretrain_hop=0,
                 g=None,eps=0,subgraphs=None,aug_type=None,subgraph_logits_ids=None,sample_num=None,sample=None):
        super(PrePrompt, self).__init__()
        self.weight=weight
        self.reduction = reduction
        self.aug_type=aug_type
        # self.lp = Lp(n_in, hidden_size)
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
            # self.model = GcnLayers(n_in, hidden_size, num_layers_num, p)
            self.model = GCN(n_in, hidden_size, 'prelu')
        if self.model_name == 'FAGCN':
            self.model = FAGCN(g,n_in, hidden_size, hidden_size, p, eps, num_layers_num).cuda()
            # self.model = FAGCN(n_in, hidden_size,hidden_size, p,eps, num_layers_num).cuda()
        if self.model_name == 'H2GCN':
            self.model = H2GCN(n_in,hidden_dim=hidden_size,class_dim=hidden_size,use_relu=False).cuda()
        #g, in_dim, hidden_dim, out_dim, dropout, eps,
        # if self.model_name == 'GraphCL':
        #     self.model_ = FAGCN(n_in, hidden_size, hidden_size, p, eps, num_layers_num).cuda() #in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2)
        #     self.model = GraphCL(hidden_size).cuda() #gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2,
        # self.read = AvgReadout()
        # self.neighborslist = neighborslist.cuda()
        # self.neighbors_2hoplist = neighbors_2hoplist.cuda()
        # self.neighborsindex = torch.LongTensor(neighborsindex).cuda()
        # self.neighbors_2hopindex = torch.LongTensor(neighbors_2hopindex).cuda()
        self.gp = gp
        self.subgraph_logits_ids = subgraph_logits_ids
        self.sample =sample
        self.subgraphs = subgraphs
        # self.self_logits_ids = self_logits_ids
        self.sample_num = sample_num
        self.sigm = nn.Sigmoid()



        # self.subadjs = subadjs
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
    #
    def make_graph(self, aug_adj1, subset_nodes):
        U1 = aug_adj1.coalesce().indices()[0]
        V1 = aug_adj1.coalesce().indices()[1]
        g1 = dgl.graph((U1, V1)).cpu()
        # g1= g1.to(aug_adj1.device)
        g1 = dgl.to_simple(g1)
        g1 = dgl.remove_self_loop(g1)
        g1 = dgl.to_bidirected(g1)
        g_ = dgl.reorder_graph(self.g, node_permute_algo='custom',
                               permute_config={'nodes_perm': subset_nodes})
        g1.ndata['d'] = g_.ndata['d']
        return g1
    # def get_neighbors(self, idx,adj):
    #
    #     neighbors = [[] for m in range(len(idx))]
    #     neighbors_2hop = [[] for m in range(len(idx))]
    #
    #     # train_adj = train_adj[:,train_range][train_range,:].A
    #     for step, x in enumerate(idx):
    #         tempneighbors, tempneighbors_2hop = process.find_2hop_neighbors(adj, idx)
    #         neighbors[step] = tempneighbors
    #         neighbors_2hop[step] = tempneighbors_2hop
    def forward1(self, seq_ori=None,adj_ori=None,seq_pos=None,adj_pos=None,seq_neg=None,adj_neg=None,edge_index=None,temperature=10,LP=False,sparse=False,
                 g_ori=None,g_pos=None,g_neg=None,
                 ori_idx=None,pos_idx=None,neg_idx=None):

        # if self.model_name == 'GAT' or self.model_name == 'GIN':
        #     logits_ = self.model(seq, edge_index)
        if self.model_name == 'FAGCN':
            # logits_origin = self.model.forward_pretrain(seq_ori,g_ori)
            # logits_pos = self.model.forward_pretrain(seq_pos, g_pos)
            logits_origin = self.model.forward_pretrain(seq_ori)
            logits_pos = self.model.forward_pretrain(seq_pos)
            # logits_neg = self.model.forward_pretrain(seq_neg, g_neg)
        elif self.model_name == 'H2GCN':
            logits_origin = self.model.forward_pretrain(adj_ori, seq_ori)
            logits_pos = self.model.forward_pretrain(adj_pos, seq_pos)
            logits_neg = self.model.forward_pretrain(adj_neg, seq_neg)
        else:
            logits_origin = self.model(seq_ori, adj_ori,sparse=sparse)
            logits_origin = self.sigm(logits_origin.squeeze(dim=0))

            logits_pos = self.model(seq_pos, adj_pos,sparse=sparse)
            logits_pos = self.sigm(logits_pos.squeeze(dim=0))

            logits_neg = self.model(seq_neg, adj_neg,sparse=sparse)
            logits_neg = self.sigm(logits_neg.squeeze(dim=0))

        # self_logits = torch_scatter.scatter(src=logits_origin, index=ori_idx, dim=0,reduce="mean")
        # pos_logits = torch_scatter.scatter(src=logits_pos, index=pos_idx, dim=0, reduce="mean")
        # neg_logits = torch_scatter.scatter(src=logits_neg, index=neg_idx, dim=0, reduce="mean")

        self_logits = torch_scatter.scatter(src=logits_origin, index=ori_idx, dim=0, reduce="sum")
        pos_logits = torch_scatter.scatter(src=logits_pos, index=pos_idx, dim=0, reduce="sum")
        # neg_logits = torch_scatter.scatter(src=logits_neg, index=neg_idx, dim=0, reduce="sum")
        #
        # neg_index = [num * self.sample_num for num in range(0, int(neg_logits.size(0) / self.sample_num))]

        self_logits = self_logits.unsqueeze(1).repeat(1,self.sample_num+1,1).cuda() #[183,1,256] -> [183,self.sample_num+1,256]

        subgraph_logits = torch.zeros((self_logits.size(0), self.sample_num+1, 256))

        # for i in range(self_logits.size(0)-1):
        #     subgraph_logits[i] = torch.cat([pos_logits[i].unsqueeze(0),neg_logits[neg_index[i]:neg_index[i+1]]],dim=0)
        for i in range(self_logits.size(0)):
            rest = [j for j in list(range(self_logits.size(0))) if j != i]
            random.shuffle(rest)
            rest= rest[: self.sample_num]
            subgraph_logits[i] = torch.cat([pos_logits[i].unsqueeze(0),pos_logits[rest]],dim=0)

        sim = F.cosine_similarity(self_logits, subgraph_logits.cuda(), dim=2)
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
        lploss = res.mean()

        return lploss
    def forward(self, seq=None, edge_index=None,adj=None,temperature=10,g=None):

        if self.model_name == 'GAT' or self.model_name == 'GIN':
            logits_ = self.model(seq, edge_index)
        elif self.model_name == 'FAGCN':
            logits_ = self.model.forward_pretrain(seq,g)
        elif self.model_name == 'H2GCN':
            logits_ = self.model.forward_pretrain(adj, seq)
        else:
            logits_ = self.model(seq, adj)
            logits_ = self.sigm(logits_.squeeze(dim=0))
        subgraph_logits = torch.zeros((logits_.size(0), self.sample_num+1, 256))
        self_logits = torch.zeros((logits_.size(0),256))
        for i in range(len(self.subgraphs)):
            tmp = logits_[self.subgraphs[i]].unsqueeze(0) if logits_[self.subgraphs[i]].dim()==1 else logits_[self.subgraphs[i]]
            if self.reduction == 'mean':
                self_logits[i] = torch.mean(tmp,0)
            if self.reduction == 'sum':
                self_logits[i] = torch.sum(tmp,0)
        self_logits = self_logits.unsqueeze(1).repeat(1,self.sample_num+1,1).cuda()
        # self_logits = self_logits.unsqueeze(1).repeat(1,self.sample_num+1,1)


        # self_logits = torch.cat([torch.mean(logits_[i].unsqueeze(0) if logits_[i].dim()==1) for i in self.subgraphs],0).unsqueeze(1).repeat(1,self.sample_num+1,1)
        # self_logits = torch.zeros((logits_.size(0), self.sample_num+1, 256))  # 2hop subgraph整合的node embedding 或者直接用最开始的logits_
        if self.aug_type == 'edge':

            for num, sample in enumerate(self.sample):
                # print('num-sample:',num)
                # subgraphs: subsets of 2hop graph nodes, sample: lists of lists of adj(一个node有长度为self.sample_num+1的adj list,第0个是增强positive，剩下的70个是其他subgrah的adj)

                for i in range(len(sample)):
                    # print(num,i)
                    # if num==1 and i==41:
                    #     print('stop')
                    tmp = logits_[self.subgraph_logits_ids[num][i]].unsqueeze(0) if type(self.subgraph_logits_ids[num][i]) == int else logits_[self.subgraph_logits_ids[num][i]]

                    if self.reduction == 'mean':
                        subgraph_logits[num][i] =torch.mean(tmp,0)
                    if self.reduction == 'sum':
                        subgraph_logits[num][i] = torch.sum(tmp, 0)
        sim = F.cosine_similarity(self_logits, subgraph_logits.cuda(), dim=2)
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
        lploss = res.mean()

        return lploss

    def embed(self, seq=None, adj=None, msk=None, LP=False, edge_index=None, g=None,sparse=False):
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        if self.model_name == 'GAT' or self.model_name == 'GIN':
            seq = torch.squeeze(seq, 0)
            h_1 = self.model(seq, edge_index)
        elif self.model_name == 'FAGCN':
            self.model.g = g
            for i in self.model.layers:
                i.g = g

            h_1 = self.model.forward_pretrain(seq)
        elif self.model_name == 'H2GCN':
            h_1 = self.model.embed(adj, seq)
        else:
            h_1 = self.model(seq, adj, sparse=sparse)
            h_1 = self.sigm(h_1.squeeze(dim=0))
        # c = self.read(h_1, msk)

        return h_1.detach() #, c.detach()
    


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


def compareloss(logits,sample,temperature,sample_adjs,sample_idx):


    h_tuples=mygather(logits,sample)
    # print("tuples",h_tuples)
    temp = torch.arange(0, len(sample))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), sample.size(1)))
    # temp = m(temp)
    temp=temp.cuda()
    h_i = mygather(logits, temp)
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

    




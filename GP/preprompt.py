import torch
import torch.nn as nn
import torch.nn.functional as F
from GP.models import DGI, GraphCL, Lp,GcnLayers
from GP.layers import GCN, AvgReadout
import tqdm
import numpy as np
def get_subgraph_3(feature, adj):
    adj_3hop = torch.matmul(adj, torch.matmul(adj, adj)).squeeze()
    #adj_3hop = torch.matmul(adj, adj).squeeze()
    adj_3hop[adj_3hop > 0] = 1  #保留距离为3以内的节点

    #print("3s adj", adj_3hop.shape)
    index = torch.nonzero(adj_3hop, as_tuple=False)
    #print("3s index", index.shape)

    
    res = torch.zeros(feature.size(0), feature.size(1)).cuda() 
    cnt = torch.zeros(feature.size(0)).cuda() 
    for i in range(index.size(0)):
        src, dst = index[i][0], index[i][1]
        res[src] += feature[dst]  # 对距离为3以内的节点的特征向量进行累加
        cnt[src] += 1
    for i in range(feature.size(0)):
        res[i] /= cnt[i]

    return res
class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, activation,a1,a2,a3,num_layers_num,p):
        super(PrePrompt, self).__init__()
        self.dgi = DGI(n_in, n_h, activation)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h,num_layers_num,p)
        self.read = AvgReadout()


        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

        # self.sample = torch.tensor(sample,dtype=int).cuda()
        # print("sample",self.sample)
        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, 
                sparse, msk, samp_bias1, samp_bias2,
                lbl,sample):
        negative_sample = torch.tensor(sample,dtype=int).cuda()
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        #logits1 = self.dgi(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        #logits2 = self.graphcledge(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
        #                           samp_bias1,
        #                           samp_bias2, aug_type='edge')
        logits3 = self.lp(self.gcn,seq1,adj,sparse)
        # print("logits1=",logits1)
        # print("lbl",lbl)
        # print("logits2=",logits2)
        # print("logits3=",logits3)
        # print("logitssize=",logits3.shape)
        #dgiloss = self.loss(logits1, lbl)
        #graphcledgeloss = self.loss(logits2, lbl)
        #sub_graph_p
        
        #print("LOGITS3",logits3.shape)
        # sub_3_feature = get_subgraph_3(logits3, adj)
        #print("sub_3_feature",sub_3_feature.shape)
        #lploss = compareloss(sub_3_feature,negative_sample,temperature=1.5)
        lploss = compareloss(logits3,negative_sample,temperature=1.5)
        lploss.requires_grad_(True)
        
        # print("promptdgi",self.dgi.prompt)
        # print("gcn",self.gcn.fc.weight)
        # print("promptLP",self.lp.prompt)


        # print("dgiloss",dgiloss)
        # print("graphcl",graphcledgeloss)
        # print("lploss",'{:.8f}'.format(lploss)) 
# 
        # print("a1=", self.a1, "a2=", self.a2,"a3=",self.a3)
        #ret =self.a1*dgiloss+self.a2*graphcledgeloss+self.a3*lploss
        ret = lploss
        return ret

    def embed(self, seq, adj, sparse, msk,LP):
        #print("seq",seq.shape)
        #print("adj",adj)
        h_1 = self.gcn(seq, adj, sparse, LP)
        h = h_1.squeeze()
        #print("h_1", h)
        # sub_3_feature = get_subgraph_3(h, adj)
        #print("sub3", sub_3_feature)
        # c = self.read(sub_3_feature, msk)
        #print("????", sub_3_feature == h)
        #return sub_3_feature.detach(), c.detach()
        return h.detach(), None
    




def mygather(feature, index):
    #print("index",index)
    #print("indexsize",index.shape)  
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    # print(tuples)

    # print("feature",feature)
    #print("featuresize",feature.shape)
    #print("index",index.shape)
    # print("indexsize",index.shape)
    res = torch.gather(feature, dim=0, index=index)
    #print("res", res.shape)
    return res.reshape(input_size,-1,feature.size(1))


def compareloss(feature,tuples,temperature):
    # print("feature",feature)
    # print("tuple",tuples)
    # feature=feature.cpu()
    # tuples = tuples.cpu()
    h_tuples=mygather(feature,tuples) #negative
    #print("tuples",h_tuples.shape)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    # temp = m(temp)
    temp=temp.cuda()  
    #print(tuples)
    #print(temp)
    h_i = mygather(feature, temp) #positive
    #print("h_i",h_i.shape)
    # print("h_tuple",h_tuples)

    #
    #readout = AvgReadout()
    #print(h_i.shape)
    #h_i_s = readout(h_i)
    #h_tuples_s = readout(h_tuples)
    #print(h_i_s.shape)
    #h_i_r = torch.sum(h_i, dim=0, keepdim=True)
    #h_tuples_r = torch.sum(h_tuples, dim=0, keepdim=True) 
    #print(h_i_r.shape)   


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



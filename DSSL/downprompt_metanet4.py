import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import GCN, AvgReadout
import torch_geometric
from promptvector import PromptVector
# from models.H2GCN.model import H2GCN
# from models.FAGCN.model import FAGCN
import torch_scatter
class downprompt(nn.Module):
    def __init__(self, neighbors, neighbors_2hop,nb_classes,feature, labels,
                 hop_level=2, reduction='mean', attention=None, multi_prompt=1, gp=0,  concat_dense=0,
                 out_size=256, bottleneck_size=64, type_vocab_size=3, pretrain_weights=None,
                 meta_in=512, dropout=0.05,hidden_size=256,activation=0,prompt=1,use_metanet=1):
        super(downprompt, self).__init__()
        # self.metanet = PromptVector(hidden_size+1, out_size, bottleneck_size)

        self.dropout = dropout
        self.activation = activation
        self.neighbor_weight = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        nn.init.xavier_normal_(self.neighbor_weight.data, gain=1.414)
        # self.neighbor_weight.data.fill_(0)
        self.hop_level = hop_level
        self.reduction = reduction
        self.attention = attention
        self.gp = gp
        self.multi_prompt = multi_prompt
        self.pretrain_weights = pretrain_weights
        self.prompt = prompt
        self.use_metanet = use_metanet
        self.t2 = nn.Linear(hidden_size, nb_classes)
        # if classifier == True:
        #     if self.model_name == 'FAGCN':
        #         self.model = FAGCN(g, n_in, hidden_size,hidden_size, p,eps, num_layers_num).cuda() #g, in_dim, hidden_dim, out_dim, dropout, eps,
        #

        # self.weight=weight
        self.concat_dense = concat_dense
        self.linear = nn.Linear(hidden_size, hidden_size)

        self.selfprompt = downstreamprompt(hidden_size).cuda()
        # self.selfprompt.weight.data = self.selfprompt.weight+ 0.1 * self.pretrain_weights.weight[0, -1]
        self.neighborsprompt = downstreamprompt(hidden_size).cuda()
        # self.neighborsprompt.weight.data = self.neighborsprompt.weight + 0.1 * self.pretrain_weights.weight[0, 0]
        self.neighbors_2hopprompt = downstreamprompt(hidden_size).cuda()
        # self.neighbors_2hopprompt.weight.data = self.neighbors_2hopprompt.weight + 0.1 * self.pretrain_weights.weight[0, 1]

        self.hidden_size = hidden_size
        self.hop_embeddings = nn.Embedding(type_vocab_size, hidden_size).cuda()
        # self.hop_embeddings = nn.Linear(type_vocab_size, hidden_size).cuda()

        self.neighbors = neighbors
        self.neighbors_2hop = neighbors_2hop

        self.nb_classes = nb_classes
        self.labels = labels
        # self.alpha = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.alpha = 1.5
        self.leakyrelu = nn.ELU()
        # self.prompt = prompt3
        # self.prompt = prompt3
        # self.a = nn.Parameter(torch.FloatTensor(1, 3), requires_grad=True).cuda()
        # self.reset_parameters()
        self.nodelabelprompt = weighted_prompt(3)

        # self.token_type_

        self.dffprompt = weighted_feature(2)

        self.embeds = feature.squeeze().cuda()
        # self.sim_matrix = torch.cosine_similarity(self.embeds.unsqueeze(1), self.embeds.unsqueeze(0), dim=-1)
        self.metanet = PromptVector(meta_in, out_size, bottleneck_size)

        # self.metanet = PromptVector(self.embeds.size(1)+self.embeds.size(0), out_size, bottleneck_size)


    def forward(self, train=0, neighbors=None, neighbors_2hop=None, idx=None):  # Model final

        center = torch.index_select(self.embeds, 0, idx)

        center_embeds = self.selfprompt(center) #[5,256]
        # center_embeds = center
        rawret = torch.zeros_like(center_embeds).cuda()#[5,256]

        # prompted_embeddings= torch.mm(origin_adj,prompts)
        for step in range(center.shape[0]):#
            if train == 1:
                # if len(self.neighbors_2hop[step]) ==0:
                #     print(step)

                tempneighbors = self.embeds[self.neighbors[step]].data #[1,256]
                # prompts_tempneighbors = prompts[self.neighbors[step]]

                tempneighbors_2hop = self.embeds[self.neighbors_2hop[step]].data
                # prompts_tempneighbors_2hop = prompts[self.neighbors_2hop[step]]

            else:
                tempneighbors = self.embeds[neighbors[step]].data
                # prompts_temp
                # neighbors = prompts[neighbors[step]]
                tempneighbors_2hop = self.embeds[neighbors_2hop[step]].data
                # prompts_tempneighbors_2hop = prompts[neighbors_2hop[step]]
            if self.prompt == 0: #use pretrained embedding
                neighborsembds = tempneighbors
                neighbors_2hopembds = tempneighbors_2hop
                center_embeds = center
            else:
                if self.multi_prompt ==1: #multi-prompts for pretrained embedding
                    neighborsembds = self.neighborsprompt(tempneighbors)
                    neighbors_2hopembds = self.neighbors_2hopprompt(tempneighbors_2hop)
                else: #single-prompts for pretrained embedding
                    neighborsembds = self.selfprompt(tempneighbors)
                    neighbors_2hopembds = self.selfprompt(tempneighbors_2hop)


            neighborsembds[neighborsembds != neighborsembds] = 0
            neighbors_2hopembds[neighbors_2hopembds != neighbors_2hopembds] = 0

            neighbor_embbedings = torch.cat((neighborsembds, neighbors_2hopembds), 0)
            # print('number of neighbors: ', neighbor_embbedings.size(0))
            self.sim_matrix = torch.cosine_similarity(center_embeds[step].unsqueeze(0).unsqueeze(1),
                                                      neighbor_embbedings.unsqueeze(0), dim=-1) #


            self.weights = F.softmax(self.sim_matrix, dim=-1)
            # self.weights = F.dropout(self.weights, self.dropout)
            if self.activation == 1:

                weighted_neighbors = F.elu(torch.mm(self.weights, neighbor_embbedings))
                center_embeddings = F.elu(center_embeds[step].unsqueeze(0))
            else:
                weighted_neighbors = torch.mm(self.weights, neighbor_embbedings) #
                center_embeddings = center_embeds[step].unsqueeze(0)

            inputs = torch.add(weighted_neighbors, center_embeddings) #【1，256】

            inputs = F.dropout(inputs, self.dropout)
            if self.use_metanet ==1:


                prompts = self.metanet(inputs)
                # rawret[step] = self.linear(inputs).squeeze(0)
                # rawret[step] = prompts.squeeze(0)
                rawret[step] = torch.add(prompts.squeeze(0),center_embeddings)
            else:
                rawret[step] = inputs.squeeze(0)

            # rawret[step] = torch.sum(self.linear(torch.cat((prompts, center_embeddings), dim=0)),0)
            # rawret[step] = torch.sum(prompts * inputs, 0)
            # rawret[step] = self.linear(torch.sum(prompts*inputs,0).unsqueeze(0)).squeeze(0)
        # rawret.require_grads=True

        if train == 1:
            self.ave = averageemb(labels=self.labels, rawret=rawret, nb_class=self.nb_classes)
        ret = torch.cosine_similarity(rawret.unsqueeze(1), self.ave.unsqueeze(0), dim=-1)
        ret = F.softmax(ret, dim=1)
        return ret
    def forward_graph1(self,neighbors=None, neighbors_2hop=None,train=1,idx=None,test_embeds=None,embeds=None):  # Model final
        if train == 1:
            if self.prompt == 1:
                center_embeds = self.selfprompt(self.embeds) #[num_graph_nodes,256]
            else:
                center_embeds = self.embeds
        else:
            if self.prompt == 1:
                center_embeds = self.selfprompt(test_embeds)
            else:
                center_embeds = test_embeds
        # center_embeds = center
        # rawret = torch.zeros((len(self.labels),self.embeds.size(1))).cuda()#[shot_graph_num,256]

        # prompted_embeddings= torch.mm(origin_adj,prompts)
        output =torch.zeros_like(center_embeds).cuda()#[36,256]
        for step in range(center_embeds.size(0)):#
            if train == 1:
                # if len(self.neighbors_2hop[step]) ==0:
                #     print(step)

                tempneighbors = self.embeds[self.neighbors[step]] #[1,256]
                # prompts_tempneighbors = prompts[self.neighbors[step]]

                tempneighbors_2hop = self.embeds[self.neighbors_2hop[step]]
                # prompts_tempneighbors_2hop = prompts[self.neighbors_2hop[step]]

            else:
                tempneighbors = embeds[neighbors[step]] #torch.index_select(embeds,0,torch.tensor(neighbors[step]).cuda())
                # prompts_temp
                # neighbors = prompts[neighbors[step]]
                tempneighbors_2hop = embeds[neighbors_2hop[step]] # torch.index_select(embeds,0,torch.tensor(neighbors_2hop[step]).cuda())
                # prompts_tempneighbors_2hop = prompts[neighbors_2hop[step]]
            if self.prompt == 0: #use pretrained embedding
                neighborsembds = tempneighbors
                neighbors_2hopembds = tempneighbors_2hop
            else:
                if self.multi_prompt ==1: #multi-prompts for pretrained embedding
                    neighborsembds = self.neighborsprompt(tempneighbors)
                    neighbors_2hopembds = self.neighbors_2hopprompt(tempneighbors_2hop)
                else: #single-prompts for pretrained embedding
                    neighborsembds = self.selfprompt(tempneighbors)
                    neighbors_2hopembds = self.selfprompt(tempneighbors_2hop)



            neighborsembds[neighborsembds != neighborsembds] = 0
            neighbors_2hopembds[neighbors_2hopembds != neighbors_2hopembds] = 0

            neighbor_embbedings = torch.cat((neighborsembds, neighbors_2hopembds), 0)
            # print('number of neighbors: ', neighbor_embbedings.size(0))
            self.sim_matrix = torch.cosine_similarity(center_embeds[step].unsqueeze(0).unsqueeze(1),
                                                      neighbor_embbedings.unsqueeze(0), dim=-1) #


            self.weights = F.softmax(self.sim_matrix, dim=-1) #[1,2]
            # self.weights = F.dropout(self.weights, self.dropout)
            if self.activation == 1:

                weighted_neighbors = F.elu(torch.mm(self.weights, neighbor_embbedings))
                center_embeddings = F.elu(center_embeds[step].unsqueeze(0))
            else:
                weighted_neighbors = torch.mm(self.weights, neighbor_embbedings) #[1,256]
                center_embeddings = center_embeds[step].unsqueeze(0)

            inputs = torch.add(weighted_neighbors, center_embeddings) #【1，256】

            inputs = F.dropout(inputs, self.dropout)
            if self.use_metanet ==1:
                prompts = self.metanet(inputs)
                # rawret[step] = self.linear(inputs).squeeze(0)
                # rawret[step] = prompts.squeeze(0)
                if center_embeddings.dim ==2:
                    center_embeddings =center_embeddings.squeeze(0)
                output[step] = torch.add(prompts.squeeze(0),center_embeddings)
            else:
                output[step]  = inputs.squeeze(0)
        if self.reduction=='mean':
            rawret = torch_scatter.scatter(src=output, index=torch.tensor(sum(idx,[])).cuda(), dim=0,reduce="mean")
        else:
            rawret = torch_scatter.scatter(src=output, index=torch.tensor(sum(idx,[])).cuda(), dim=0, reduce="sum")

            # rawret[step] = torch.sum(self.linear(torch.cat((prompts, center_embeddings), dim=0)),0)
            # rawret[step] = torch.sum(prompts * inputs, 0)
            # rawret[step] = self.linear(torch.sum(prompts*inputs,0).unsqueeze(0)).squeeze(0)
        # rawret.require_grads=True

        rawret = self.t2(rawret)
        rawret = F.log_softmax(rawret, 1)
        return rawret
    def forward_graph(self,neighbors=None, neighbors_2hop=None,train=1,idx=None,test_embeds=None,embeds=None):  # Model final
        if train == 1:
            if self.prompt == 1:
                center_embeds = self.selfprompt(self.embeds) #[num_graph_nodes,256]
            else:
                center_embeds = self.embeds
        else:
            if self.prompt == 1:
                center_embeds = self.selfprompt(test_embeds)
            else:
                center_embeds = test_embeds
        # center_embeds = center
        # rawret = torch.zeros((len(self.labels),self.embeds.size(1))).cuda()#[shot_graph_num,256]

        # prompted_embeddings= torch.mm(origin_adj,prompts)
        output =torch.zeros_like(center_embeds).cuda()#[36,256]
        for step in range(center_embeds.size(0)):#
            if train == 1:
                # if len(self.neighbors_2hop[step]) ==0:
                #     print(step)

                tempneighbors = self.embeds[self.neighbors[step]] #[1,256]
                # prompts_tempneighbors = prompts[self.neighbors[step]]

                tempneighbors_2hop = self.embeds[self.neighbors_2hop[step]]
                # prompts_tempneighbors_2hop = prompts[self.neighbors_2hop[step]]

            else:
                tempneighbors = embeds[neighbors[step]] #torch.index_select(embeds,0,torch.tensor(neighbors[step]).cuda())
                # prompts_temp
                # neighbors = prompts[neighbors[step]]
                tempneighbors_2hop = embeds[neighbors_2hop[step]] # torch.index_select(embeds,0,torch.tensor(neighbors_2hop[step]).cuda())
                # prompts_tempneighbors_2hop = prompts[neighbors_2hop[step]]
            if self.prompt == 0: #use pretrained embedding
                neighborsembds = tempneighbors
                neighbors_2hopembds = tempneighbors_2hop
            else:
                if self.multi_prompt ==1: #multi-prompts for pretrained embedding
                    neighborsembds = self.neighborsprompt(tempneighbors)
                    neighbors_2hopembds = self.neighbors_2hopprompt(tempneighbors_2hop)
                else: #single-prompts for pretrained embedding
                    neighborsembds = self.selfprompt(tempneighbors)
                    neighbors_2hopembds = self.selfprompt(tempneighbors_2hop)



            neighborsembds[neighborsembds != neighborsembds] = 0
            neighbors_2hopembds[neighbors_2hopembds != neighbors_2hopembds] = 0

            neighbor_embbedings = torch.cat((neighborsembds, neighbors_2hopembds), 0)
            # print('number of neighbors: ', neighbor_embbedings.size(0))
            self.sim_matrix = torch.cosine_similarity(center_embeds[step].unsqueeze(0).unsqueeze(1),
                                                      neighbor_embbedings.unsqueeze(0), dim=-1) #


            self.weights = F.softmax(self.sim_matrix, dim=-1) #[1,2]
            # self.weights = F.dropout(self.weights, self.dropout)
            if self.activation == 1:

                weighted_neighbors = F.elu(torch.mm(self.weights, neighbor_embbedings))
                center_embeddings = F.elu(center_embeds[step].unsqueeze(0))
            else:
                weighted_neighbors = torch.mm(self.weights, neighbor_embbedings) #[1,256]
                center_embeddings = center_embeds[step].unsqueeze(0)

            inputs = torch.add(weighted_neighbors, center_embeddings) #【1，256】

            inputs = F.dropout(inputs, self.dropout)
            if self.use_metanet ==1:
                prompts = self.metanet(inputs)
                # rawret[step] = self.linear(inputs).squeeze(0)
                # rawret[step] = prompts.squeeze(0)
                if center_embeddings.dim ==2:
                    center_embeddings =center_embeddings.squeeze(0)
                output[step] = torch.add(prompts.squeeze(0),center_embeddings)
            else:
                output[step]  = inputs.squeeze(0)
        if self.reduction=='mean':
            rawret = torch_scatter.scatter(src=output, index=torch.tensor(sum(idx,[])).cuda(), dim=0,reduce="mean")
        else:
            rawret = torch_scatter.scatter(src=output, index=torch.tensor(sum(idx,[])).cuda(), dim=0, reduce="sum")

            # rawret[step] = torch.sum(self.linear(torch.cat((prompts, center_embeddings), dim=0)),0)
            # rawret[step] = torch.sum(prompts * inputs, 0)
            # rawret[step] = self.linear(torch.sum(prompts*inputs,0).unsqueeze(0)).squeeze(0)
        # rawret.require_grads=True

        if train == 1:
            self.ave = averageemb(labels=self.labels, rawret=rawret, nb_class=self.nb_classes)
        ret = torch.cosine_similarity(rawret.unsqueeze(1), self.ave.unsqueeze(0), dim=-1)
        ret = F.softmax(ret, dim=1)
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


def averageemb(labels, rawret, nb_class):
    retlabel = torch.FloatTensor(nb_class, int(rawret.shape[0] / nb_class), int(rawret.shape[1])).cuda()
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    cnt7 = 0
    # print("labels",labels)
    for x in range(0, rawret.shape[0]):
        if labels[x].item() == 0:
            retlabel[0][cnt1] = rawret[x]
            cnt1 = cnt1 + 1
        if labels[x].item() == 1:
            retlabel[1][cnt2] = rawret[x]
            cnt2 = cnt2 + 1
        if labels[x].item() == 2:
            retlabel[2][cnt3] = rawret[x]
            cnt3 = cnt3 + 1
        if labels[x].item() == 3:
            retlabel[3][cnt4] = rawret[x]
            cnt4 = cnt4 + 1
        if labels[x].item() == 4:
            retlabel[4][cnt5] = rawret[x]
            cnt5 = cnt5 + 1
        if labels[x].item() == 5:
            retlabel[5][cnt6] = rawret[x]
            cnt6 = cnt6 + 1
        if labels[x].item() == 6:
            retlabel[6][cnt7] = rawret[x]
            cnt7 = cnt7 + 1
    retlabel = torch.mean(retlabel, dim=1)
    return retlabel


class weighted_prompt(nn.Module):
    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(1)
        self.weight[0][1].data.fill_(0.1)
        self.weight[0][2].data.fill_(30)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding = torch.mm(self.weight, graph_embedding)
        return graph_embedding


class weighted_feature(nn.Module):
    def __init__(self, weightednum):
        super(weighted_feature, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(1)
        self.weight[0][1].data.fill_(0)

    def forward(self, graph_embedding1, graph_embedding2):
        # print("weight",self.weight)
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)


class downstreamprompt(nn.Module):
    def __init__(self, hid_units):
        super(downstreamprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0.3)
        # self.weight[0][1].data.fill_(0.3)
        # self.weight[0][2].data.fill_(0.3)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding = self.weight * graph_embedding
        return graph_embedding


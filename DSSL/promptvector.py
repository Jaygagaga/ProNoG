import torch
import torch.nn as nn
import torch.nn.functional as F


# class PromptVector(nn.Module):
#     def __init__(self,hid_units, hop_level):
#         super(PromptVector, self).__init__()
#         self.weight= nn.Parameter(torch.FloatTensor(hop_level+1,hid_units), requires_grad=True) #[1,256]
#         self.act = nn.ELU()
#         self.reset_parameters()
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight)
#
#         # self.weight[0][0].data.fill_(0.3)
#         # self.weight[0][1].data.fill_(0.3)
#         # self.weight[0][2].data.fill_(0.3)
#     def forward(self, graph_embedding):
#         # print("weight",self.weight)
#         if graph_embedding.dim() == 3:
#             graph_embedding=self.weight.unsqueeze(1).repeat(1,graph_embedding.size(1),1)*graph_embedding
#         else:
#             graph_embedding = self.weight*graph_embedding
#         return graph_embedding


# class PromptVector(nn.Module): #生成输入一个subgraph（anchornodes+hop1+hop2），生成对应数量的prompts
#     def __init__(self,in_size,out_size,bottleneck_size,bottleneck_dropout=0.1,scaling=0.1):
#         super(PromptVector, self).__init__()
#         self.down = nn.Linear(in_size, bottleneck_size, bias=True) #
#         self.up = nn.Linear(bottleneck_size, out_size, bias=True)
#         # self.weight = nn.Parameter(torch.FloatTensor(prompt_num,embedding_size), requires_grad=True)
#         self.scaling=scaling
#         self.act_fn = nn.Tanh()
#         self.dropout=nn.Dropout(p=bottleneck_dropout)
#         self.reset_parameters()
#
#         # self.device = device
#
#     def reset_parameters(self):
#         # conv1.weight.data.fill_(0.01)
#         torch.nn.init.xavier_uniform_(self.down.weight)
#         torch.nn.init.xavier_uniform_(self.up.weight)
#         # pass
#     def forward(self,node_embeddings): #node_embeddings (anchornodes+hop1+hop2)[1,num_nodes*embedding_size]->[1,bottleneck_size]-> [1,embedding_size]
#         if self.scaling:
#             prompts = self.up(self.act_fn(self.down(self.dropout(node_embeddings)))).to(node_embeddings.device) * self.scaling
#         else:
#             prompts = self.up(self.act_fn(self.down(self.dropout(node_embeddings)))).to(node_embeddings.device) #[3,embedding_size]
#
#         return prompts #[1,embedding_size] [1,183]
class PromptVector(nn.Module): #生成输入一个subgraph（anchornodes+hop1+hop2），生成对应数量的prompts
    def __init__(self,in_size,out_size,bottleneck_size,bottleneck_dropout=0.1,scaling=0.1):
        super(PromptVector, self).__init__()
        self.down = nn.Linear(in_size, bottleneck_size, bias=True) #
        self.up = nn.Linear(bottleneck_size, out_size, bias=True)
        # self.weight = nn.Parameter(torch.FloatTensor(prompt_num,embedding_size), requires_grad=True)
        self.scaling=scaling
        self.act_fn = nn.Tanh()
        self.dropout=nn.Dropout(p=bottleneck_dropout)
        self.reset_parameters()

        # self.device = device

    def reset_parameters(self):
        # conv1.weight.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.down.weight)
        torch.nn.init.xavier_uniform_(self.up.weight)
        # pass
    def forward(self,node_embeddings,sim_matrix=None): #node_embeddings [183,256+183]->[183, 64]-> [183, 256]
        if sim_matrix != None:
            node_embeddings_ = torch.cat((node_embeddings, sim_matrix), dim=1)
        else:
            node_embeddings_ = node_embeddings
        if self.scaling:
            prompts = self.up(self.act_fn(self.down(self.dropout(node_embeddings_)))).to(node_embeddings_.device) * self.scaling
        else:
            prompts = self.up(self.act_fn(self.down(self.dropout(node_embeddings_)))).to(node_embeddings_.device) #[3,embedding_size]

        return prompts # [183,256]
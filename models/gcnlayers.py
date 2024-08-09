import torch

from layers import GCN, AvgReadout
import torch.nn as nn
class downstreamprompt(nn.Module):
    def __init__(self,hid_units):
        super(downstreamprompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0.3)
        # self.weight[0][1].data.fill_(0.3)
        # self.weight[0][2].data.fill_(0.3)
    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding=self.weight * graph_embedding
        return graph_embedding
class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h,num_layer=2,dropout=0.05,GPextension=True,alpha=None):
        super(GcnLayers, self).__init__()

        self.act=torch.nn.ReLU()
        self.num_layer=num_layer
        self.g_net, self.bns = self.create_net(n_in,n_h,self.num_layer)
        self.GPextension= GPextension
        if GPextension:
            self.prompt1 = downstreamprompt(n_in)
            self.prompt2 = downstreamprompt(n_h)
            # self.prompt3 = downstreamprompt(n_h)
            self.alpha = alpha


        self.dropout=torch.nn.Dropout(p=dropout)

    def create_net(self,input_dim, hidden_dim,num_layers):

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                nn = GCN(hidden_dim, hidden_dim)
            else:
                nn = GCN(input_dim, hidden_dim)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns


    def forward(self, seq, adj,LP=True,GPextension=False):
        graph_output = torch.squeeze(seq,dim=0)
        if GPextension:
            graph_output1 = self.prompt1(graph_output)
            graph_output2 = graph_output
            # for i in range(self.num_layer):
            #     # print("i",i)
            #     input = (graph_output1, adj)
            #     graph_output1 = self.convs[i](input)
            #     if self.GPextension and i == 0:
            #         graph_output1 = self.prompt2(graph_output1)
            #     if self.GPextension and i == 1:
            #         graph_output1 = self.prompt3(graph_output1)
            #     # print("graphout1",graph_output)
            #     # print("graphout1",graph_output.shape)
            #     if LP:
            #         # print("graphout1",graph_output.shape)
            #         graph_output1 = self.bns[i](graph_output1)
            #         # print("graphout2",graph_output.shape)
            #         graph_output1 = self.dropout(graph_output1)
            # return graph_output1.unsqueeze(dim=0)
            for i in range(self.num_layer):
                # print("i",i)
                input = (graph_output1, adj)
                graph_output1 = self.convs[i](input)
                # print("graphout1",graph_output)
                # print("graphout1",graph_output.shape)
                if LP:
                    # print("graphout1",graph_output.shape)
                    graph_output1 = self.bns[i](graph_output1)
                    # print("graphout2",graph_output.shape)
                    graph_output1 = self.dropout(graph_output1)
            for i in range(self.num_layer):
                # print("i",i)
                input = (graph_output2, adj)
                graph_output2 = self.convs[i](input)
                # if self.GPextension and i == 0:
                #     graph_output2 = self.prompt2(graph_output)
                if self.GPextension and i == self.num_layer-1:
                    graph_output2 = self.prompt2(graph_output2)
                # print("graphout1",graph_output)
                # print("graphout1",graph_output.shape)
                if LP:
                    # print("graphout1",graph_output.shape)
                    graph_output2 = self.bns[i](graph_output2)
                    # print("graphout2",graph_output.shape)
                    graph_output2 = self.dropout(graph_output2)
            embedding = self.alpha*graph_output1 + graph_output2
            return embedding.unsqueeze(dim=0)


        else:
            for i in range(self.num_layer):
                # print("i",i)
                input=(graph_output,adj)
                graph_output = self.convs[i](input)
                # print("graphout1",graph_output)
                # print("graphout1",graph_output.shape)
                if LP:
                    # print("graphout1",graph_output.shape)
                    graph_output = self.bns[i](graph_output)
                    # print("graphout2",graph_output.shape)
                    graph_output = self.dropout(graph_output)

            # print("Xs",xs)

            return graph_output.unsqueeze(dim=0)

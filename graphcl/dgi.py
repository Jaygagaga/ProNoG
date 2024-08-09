import torch
import torch.nn as nn
from graphcl.gcn import GCN
from graphcl.discriminator import Discriminator
from graphcl.readout import AvgReadout
import pdb
from models.FAGCN.model import FAGCN
class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation,model_name='GCN',g=None,eps=0,p=0.05):
        super(DGI, self).__init__()
        self.model_name = model_name
        if model_name == 'GCN':
            self.gcn = GCN(n_in, n_h, activation)
        if model_name == 'FAGCN':
            self.gcn = FAGCN(g, n_in, n_h, n_h, p, eps, 2).cuda()
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

        # self.disc2 = Discriminator2(n_h)

    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2, aug_type,
                g1=None,g2=None,g=None
                ):
        if self.model_name == 'GCN':
            h_0 = self.gcn(seq1, adj, sparse)
            if aug_type == 'edge':

                h_1 = self.gcn(seq1, aug_adj1, sparse)
                h_3 = self.gcn(seq1, aug_adj2, sparse)

            # elif aug_type == 'mask':
            #
            #     h_1 = self.gcn(seq3, adj, sparse)
            #     h_3 = self.gcn(seq4, adj, sparse)
            #
            # elif aug_type == 'node' or aug_type == 'subgraph':
            #
            #     h_1 = self.gcn(seq3, aug_adj1, sparse)
            #     h_3 = self.gcn(seq4, aug_adj2, sparse)
            #
            # else:
            #     assert False

            c_1 = self.read(h_1, msk)
            c_1= self.sigm(c_1)

            c_3 = self.read(h_3, msk)
            c_3= self.sigm(c_3)

            h_2 = self.gcn(seq2, adj, sparse)

            ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
            ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)

            ret = ret1 + ret2
            return ret
        if self.model_name == 'FAGCN':
            for i in self.gcn.layers:
                i.g = g
            h_0 = self.gcn.forward_pretrain(seq1,g)
            h_2 = self.gcn.forward_pretrain(seq2,g)
            h_0 = h_0.unsqueeze(0)
            h_2 = h_2.unsqueeze(0)
            if aug_type == 'edge':
                for i in self.gcn.layers:
                    i.g1 = g1
                h_1 = self.gcn.forward_pretrain(seq1, g1)
                for i in self.gcn.layers:
                    i.g2 = g2
                h_3 = self.gcn.forward_pretrain(seq1, g2)
            h_1 = h_1.unsqueeze(0)
            h_3 = h_3.unsqueeze(0)

            c_1 = self.read(h_1, msk)
            c_1 = self.sigm(c_1)

            c_3 = self.read(h_3, msk)
            c_3 = self.sigm(c_3)


            ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
            ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)

            ret = ret1 + ret2
            return ret


    # Detach the return variables
    def embed(self, seq, adj=None, sparse=None, msk=None,g=None):
        if self.model_name == 'GCN':
            h_1 = self.gcn(seq, adj, sparse)
            # h_1 = self.sigm(h_1)
            c = self.read(h_1, msk)
        if self.model_name == 'FAGCN':
            for i in self.gcn.layers:
                i.g = g
            h_1 = self.gcn.forward_pretrain(seq,g)

        return h_1.detach(), None


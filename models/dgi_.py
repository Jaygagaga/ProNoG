"""
Deep Graph Infomax in DGL

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""

import math

import torch
import torch.nn as nn
from models.gcn_dgi import GCN


class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(
            g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout
        )

    def forward(self, features, g=None,corrupt=False):
        if g!=None:
            if corrupt:
                perm = torch.randperm(g.num_nodes())
                features = features[perm]
            self.conv.g = g
            features = self.conv(features)
        else:
            if corrupt:
                perm = torch.randperm(self.g.num_nodes())
                features = features[perm]
            features = self.conv(features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(DGI, self).__init__()
        self.encoder = Encoder(
            g, in_feats, n_hidden, n_layers, activation, dropout
        )
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features,g=None):
        positive = self.encoder(features, corrupt=False,g=g)
        negative = self.encoder(features, corrupt=True,g=g)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2
    def forward_embed(self,features):
        positive = self.encoder(features, corrupt=False)
        return positive
class DGI1(nn.Module):
    def __init__(self,g,in_feats, n_hidden, n_layers, activation, dropout):
        super(DGI1, self).__init__()
        self.encoder = Encoder(g,in_feats, n_hidden, n_layers, activation, dropout
        )
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features,g):
        positive = self.encoder(features, g,corrupt=False)
        negative = self.encoder(features,g, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2
    def forward_embed(self,features):
        positive = self.encoder(features, corrupt=False)
        return positive

class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)

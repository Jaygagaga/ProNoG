import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp
from layers import GAT, AvgReadout
import tqdm
import numpy as np

class GATLayers(nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features, num_heads, dropout=0.0, alpha=0.2):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GAT(in_features, hidden_features, num_heads, dropout, alpha))
        for _ in range(num_layers - 1):
            self.layers.append(GAT(hidden_features, hidden_features, num_heads, dropout, alpha))
        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x, adj):
        for layer in self.layers:
            x = F.elu(layer(x, adj))
        x = self.fc(x)
        return x

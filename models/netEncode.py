import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import torch
import torch.nn.functional as F
from collections import OrderedDict
import config as cfg
import numpy as np


class MSA(nn.Module):
    def __init__(self, in_feature=64, out_feature=128,
                 hidden_dim=128, num_layer=2,
                 alpha=0.1, activator=None):
        super(MSA, self).__init__()
        self.gcns = torch.nn.ModuleList()

        for i in range(num_layer-1):
            if i == 0:  # Agg-GTA
                gcn = GCNConv(in_feature, hidden_dim)
            else:
                gcn = GCNConv(hidden_dim, hidden_dim)
            self.gcns.append(gcn)

        self.gat = GATConv(in_feature, hidden_dim, dropout=0.6)  # Agg-LTA
        self.w_gcn = nn.Parameter(self.init_randn(hidden_dim, out_feature))
        self.w_gat = nn.Parameter(self.init_randn(hidden_dim, out_feature))
        self.alpha = alpha
        self.drop = nn.Dropout(0.6)
        if activator is None:
            self.activator = nn.Tanh()
        else:
            self.activator = activator

    @staticmethod
    def init_randn(size, dim):
        emb = nn.Parameter(torch.randn(size, dim))
        emb.data = F.normalize(emb.data)
        return emb

    def forward(self, x, edge_index):
        # @edge_index: shape(2,num_edge), first row: source，second row: target
        x_gcn = x
        for i in range(len(self.gcns)):
            x_gcn = self.drop(self.gcns[i](x_gcn, edge_index))

        x_gat = self.gat(x, edge_index)

        x = x_gcn @ self.w_gcn + self.alpha * x_gat @ self.w_gat  # Eq.10
        x = self.drop(x)
        return self.activator(x)


class NetEncode(nn.Module):
    def __init__(self):
        super(NetEncode, self).__init__()
        self.msas = nn.Sequential(OrderedDict([
            (
            'MSA1', MSA(in_feature=cfg.dim_feature, hidden_dim=128,
                        out_feature=cfg.msa_out_dim, num_layer=cfg.num_layer)),
            # ('MSA2', MSA(64, 64)),
            # ('MSA3', MSA(64, 64)),
        ]))

    def forward(self, x, edge_index):
        ys = [x]
        for i in range(len(self.msas)):
            ys.append(self.msas[i](ys[i], edge_index))
        return torch.cat(ys, dim=1)


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


if __name__ == '__main__':
    # model = NetEncode(in_feature=cfg.dim_feature, hidden_dim=256, out_feature=cfg.msa_out_dim)
    model = NetEncode()
    print(params_count(model))




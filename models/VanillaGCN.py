# code for Vanilla GCN model
#
# using PyTorch Geometric (PyG) GCNConv

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import tqdm
import gc
import numpy as np
from BaseGNN import baseGNN, base_set_up_model, train, test, predict

class VanillaGCN(baseGNN):
    def __init__(self, node_features, edge_features, num_classes, num_hidden=5, num_layers=3, dropout=0.2):
        super(VanillaGCN, self).__init__(node_features, edge_features, num_classes, node_features, num_layers, dropout)
        self.convs.append(GCNConv(node_features, num_hidden))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(num_hidden, num_hidden))
        self.convs.append(GCNConv(num_hidden, num_classes))

def set_up_model(dataset, num_classes, device, loss_type="mse"):
    num_classes, criterion = base_set_up_model(dataset, num_classes, device, loss_type)
    model = VanillaGCN(dataset.num_features, dataset.num_edge_features, num_classes, num_hidden = 5, num_layers=3, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer, criterion
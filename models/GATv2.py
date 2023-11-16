# code for a basic Graph Attention Network (GAT) model
# for graph label regression
# GATv2 == https://arxiv.org/pdf/2105.14491.pdf
# using PyTorch Geometric (PyG) GATv2Conv
#
# note that our graphs have 1 node feature (the pixel value)
# and 2 edge features (the separation and angle between the two pixels)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
from torch.nn import Linear
from torch_geometric.nn import aggr
import tqdm, gc, numpy as np
from BaseGNN import baseGNN, base_set_up_model, train, test, predict

class GATv2(baseGNN):
    def __init__(self, node_features, edge_features, num_classes, \
                 hidden_channels=8, num_layers=3, heads=8, dropout=0, negative_slope=0.2):
        super(GATv2, self).__init__(node_features, edge_features, num_classes, hidden_channels * heads,\
              num_layers, dropout, negative_slope)
        self.hidden_channels = hidden_channels
        self.heads = heads
        
        self.convs.append(GATv2Conv(node_features, hidden_channels, \
                            edge_dim=edge_features, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, \
                            edge_dim=edge_features, heads=heads, dropout=dropout))

# look at this function for setting hyperparameters, runGNN uses this -->
def set_up_model(dataset, num_classes, device, loss_type="L1Loss"):
    num_classes, criterion = base_set_up_model(dataset, num_classes, device, loss_type)
    model = GATv2(dataset.num_features, dataset.num_edge_features, num_classes, \
                        hidden_channels=8, dropout=0.1, negative_slope=0.2, \
                        num_layers=4, heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer, criterion
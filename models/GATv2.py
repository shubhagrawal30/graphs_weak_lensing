# code for a basic Graph Attention Network (GAT) model
# for graph label regression
# GATv2 == https://arxiv.org/pdf/2105.14491.pdf
# using PyTorch Geometric (PyG) GATv2Conv
#
# note that our graphs have 1 node feature (the pixel value)
# and 2 edge features (the separation and angle between the two pixels)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear

class GATv2(torch.nn.Module):
    def __init__(self, node_features, edge_features, num_classes, \
                 hidden_channels=64, num_layers=3, heads=8, dropout=0.1, negative_slope=0.2):
        super(GATv2, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(node_features, hidden_channels, \
                            edge_dim=edge_features, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, \
                            edge_dim=edge_features, heads=heads, dropout=dropout))
        self.linear = Linear(hidden_channels * heads, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index.to(torch.int64), data.edge_attr
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.linear(x)
        return x
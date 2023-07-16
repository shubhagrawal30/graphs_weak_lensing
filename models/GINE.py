# code for a Graph Isomorphism Network (GIN) model
# that also consider edge features
# for graph label regression
# GIN == https://arxiv.org/pdf/1810.00826.pdf
# using PyTorch Geometric (PyG) GINEConv
#
# note that our graphs have 1 node feature (the pixel value)
# and 2 edge features (the separation and angle between the two pixels)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from torch.nn import Linear
from torch_geometric.nn import aggr

class GINE(torch.nn.Module):
    def __init__(self, node_features, edge_features, num_classes, \
                 num_layers=3, dropout=0, negative_slope=0.2, \
                 internal_nn_layers=2, internal_nn_hidden_channels=64):
        super(GINE, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_classes = num_classes
        self.internal_nn_layers = internal_nn_layers
        self.internal_nn_hidden_channels = internal_nn_hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.node_norm = BatchNorm(node_features)
        self.edge_norm = BatchNorm(edge_features)
        
        self.internal_nn = lambda inp, out: torch.nn.Sequential(\
            Linear(inp, internal_nn_hidden_channels), torch.nn.ReLU(), \
                Linear(internal_nn_hidden_channels, out))
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINEConv(self.internal_nn(node_features, \
                        internal_nn_hidden_channels), train_eps=True, edge_dim=edge_features))
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(self.internal_nn(internal_nn_hidden_channels, \
                        internal_nn_hidden_channels), train_eps=True, edge_dim=edge_features))
        
        self.readout = aggr.SoftmaxAggregation(learn=True)
        self.linear = Linear(internal_nn_hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index.to(torch.int64), data.edge_attr
        x, edge_attr = self.node_norm(x), self.edge_norm(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.readout(x, data.batch)
        x = self.linear(x)
        return x
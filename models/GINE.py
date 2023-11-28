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
from torch.nn import Linear, BatchNorm1d, ReLU
from torch_geometric.nn import aggr
import tqdm, gc, numpy as np
from BaseGNN import baseGNN, base_set_up_model, train, test, predict

class GINE(baseGNN):
    def __init__(self, node_features, edge_features, num_classes, \
                 num_layers=3, dropout=0, negative_slope=0.2, \
                 internal_nn_layers=2, internal_nn_hidden_channels=64):
        super(GINE, self).__init__(node_features, edge_features, num_classes, 4*internal_nn_hidden_channels,\
              num_layers, dropout, negative_slope)
        self.internal_nn_layers = internal_nn_layers # currently not used TODO!
        self.internal_nn_hidden_channels = inhc = internal_nn_hidden_channels 
        
        self.internal_nn = lambda inp, out: torch.nn.Sequential(\
            Linear(inp, inhc), BatchNorm1d(inhc), ReLU(), \
                Linear(inhc, out), ReLU(),)
        
        self.convs.append(GINEConv(self.internal_nn(node_features, \
                        inhc), train_eps=True, edge_dim=edge_features))
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(self.internal_nn(inhc, \
                        inhc), train_eps=True, edge_dim=edge_features))

def set_up_model(dataset, num_classes, device, loss_type="L1Loss"):
    num_classes, criterion = base_set_up_model(dataset, num_classes, device, loss_type)
    model = GINE(dataset.num_features, dataset.num_edge_features, num_classes, \
                        num_layers=4, dropout=0.2, negative_slope=0.2, \
                        internal_nn_layers=None, internal_nn_hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, optimizer, criterion
    
# code for a "Heterogenous "Graph Isomorphism Network (GIN) model
# that also consider edge features
# for graph label regression
# GIN == https://arxiv.org/pdf/1810.00826.pdf
# using PyTorch Geometric (PyG) GINEConv
#
# note that our graphs have 1 node feature (the pixel value)
# and 2 edge features (the separation and angle between the two pixels)
#
# heterogenous = each datapoint is several different graphs

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from torch.nn import Linear, BatchNorm1d, ReLU
from torch_geometric.nn import aggr
import tqdm, gc, numpy as np
from BaseGNN import baseGNN, base_set_up_model, train, test, predict

class HetGINE(baseGNN):
    def __init__(self, node_features, edge_features, num_classes, num_subgraphs, \
                num_layers=3, dropout=0, negative_slope=0.2, \
                internal_nn_layers=2, internal_nn_hidden_channels=64):
        super(HetGINE, self).__init__(node_features, edge_features, num_classes, 
                    num_subgraphs*num_layers*internal_nn_hidden_channels, num_layers, dropout, negative_slope)
        self.internal_nn_layers = internal_nn_layers # currently not used TODO!
        self.internal_nn_hidden_channels = inhc = internal_nn_hidden_channels 
        self.num_subgraphs = num_subgraphs
        
        self.internal_nn = lambda inp, out: torch.nn.Sequential(\
            Linear(inp, inhc), BatchNorm1d(inhc), ReLU(), \
                Linear(inhc, out), ReLU(),)
        
        self.convs.append(GINEConv(self.internal_nn(node_features, \
                        inhc), train_eps=True, edge_dim=edge_features))
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(self.internal_nn(inhc, \
                        inhc), train_eps=True, edge_dim=edge_features))
            
    def forward(self, data):
        all_h = []
        for dpt in data:
            x, edge_index, edge_attr = dpt.x, dpt.edge_index.to(torch.int64), dpt.edge_attr
            x, edge_attr = self.node_norm(x), self.edge_norm(edge_attr)
            resx = {}
            for ind, conv in enumerate(self.convs):
                x = conv(x, edge_index, edge_attr)
                x = F.dropout(x, p=self.dropout, training=self.training)
                resx[ind] = torch.clone(x)
                resx[ind] = self.readout(resx[ind], dpt.batch)
            h = torch.cat([resx[i] for i in range(len(self.convs))], dim=1)
            all_h.append(h)
        h = torch.cat(all_h, dim=1)
        for linear in self.linears:
            h = F.leaky_relu(linear(h), negative_slope=self.negative_slope)
            h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.linear_fin(h)
        return x

def set_up_model(dataset, num_classes, device, num_subgraphs, loss_type="L1Loss"):
    num_classes, criterion = base_set_up_model(dataset, num_classes, device, loss_type)
    model = HetGINE(dataset.num_features, dataset.num_edge_features, num_classes, num_subgraphs, \
                        num_layers=3, dropout=0.2, negative_slope=0.2, \
                        internal_nn_layers=None, internal_nn_hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, optimizer, criterion

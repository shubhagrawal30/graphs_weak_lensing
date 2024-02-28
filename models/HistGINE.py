# code for a "Histogram-based" Graph Isomorphism Network (GIN) model
# that also consider edge features
# for graph label regression
# GIN == https://arxiv.org/pdf/1810.00826.pdf
# using PyTorch Geometric (PyG) GINEConv
#
# note that our graphs have 1 node feature (the pixel value)
# and 2 edge features (the separation and angle between the two pixels)
#
# heterogenous = each datapoint is several different graphs
# histogram based = we will essentially use the histogram NN with some graph convolutions

import torch, torchist
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from torch.nn import Linear, BatchNorm1d, ReLU
from torch_geometric.nn import aggr
from torch_geometric.utils import unbatch
import tqdm, gc, numpy as np
from BaseGNN import baseGNN, base_set_up_model, train, test, predict

import matplotlib.pyplot as plt

class HistGINE(baseGNN):
    def __init__(self, node_features, edge_features, num_classes, num_subgraphs, scales, tomobins, \
                num_layers=3, dropout=0, negative_slope=0.2, nbins=14, \
                internal_nn_layers=2, internal_nn_hidden_channels=64):
        super(HistGINE, self).__init__(node_features, edge_features, num_classes, 
                    num_subgraphs*num_layers*internal_nn_hidden_channels, num_layers, dropout, negative_slope)
        self.internal_nn_layers = internal_nn_layers # currently not used TODO!
        self.internal_nn_hidden_channels = inhc = internal_nn_hidden_channels 
        self.num_subgraphs = num_subgraphs
        
        self.internal_nn = lambda inp, out: torch.nn.Sequential(\
            Linear(inp, inhc), BatchNorm1d(inhc), ReLU(), \
                Linear(inhc, out), ReLU(),)
        
        self.convs_list = torch.nn.ModuleList()
        for i in range(num_subgraphs):
            self.convs_list.append(torch.nn.ModuleList())
            self.convs_list[i].append(GINEConv(self.internal_nn(node_features, \
                            inhc), train_eps=True, edge_dim=edge_features))
            for _ in range(num_layers - 1):
                self.convs_list[i].append(GINEConv(self.internal_nn(inhc, \
                            inhc), train_eps=True, edge_dim=edge_features))
                
        self.scales = scales
        self.tomobins = tomobins
        self.nbins = 14
        self.compute_bin_extrema()
        
        # self.hinp = hinp = nbins*len(scales)*len(tomobins)*inhc
        self.hinp = hinp = nbins*len(scales)*len(tomobins)*(num_layers+1)
        self.layers = torch.nn.ModuleList()
        self.layers.append(Linear(hinp, 128))
        # for i in range(num_dense_layers - 2):
        #     self.layers.append(Linear(dense_layer_size, dense_layer_size))
        self.layers.append(Linear(128, 128))
        self.layers.append(Linear(128, 128))
        self.layers.append(Linear(128, 64))
        self.layers.append(Linear(64, 64))
        self.layers.append(Linear(64, 32))
        self.layers.append(Linear(32, num_classes))
        self.input_norm = BatchNorm1d(hinp)
        
    def get_sm_t_idx(self, smi, ti):
        return smi + ti * len(self.scales)
        
    def compute_bin_extrema(self, \
            ex_file='/pscratch/sd/s/shubh/graph_data/dirac_peaks_extrema.npy'):
        ex = np.load(ex_file, allow_pickle=True).item()
        bins = np.empty((len(self.scales)*len(self.tomobins), self.nbins+1))
        for smi, sm in enumerate(self.scales):
            for ti, t in enumerate(self.tomobins):
                bins[self.get_sm_t_idx(smi, ti)] = np.linspace(ex[t][sm][0],ex[t][sm][1], self.nbins+1)
        self.bins = torch.tensor(bins).to(torch.float32)
    
    def forward(self, data):
        resx = {}
        for sm in range(len(self.scales)):
            for t in range(len(self.tomobins)):
                smtidx = self.get_sm_t_idx(sm, t)
                dpt = data[smtidx]
                x, edge_index, edge_attr = dpt.x, dpt.edge_index.to(torch.int64), dpt.edge_attr
                # x, edge_attr = self.node_norm(x), self.edge_norm(edge_attr)
                resx[smtidx] = {}
                resx[smtidx][0] = unbatch(x, dpt.batch)
                for ind, conv in enumerate(self.convs_list[smtidx]):
                    x = conv(x, edge_index, edge_attr)
                    # x = F.leaky_relu(x, negative_slope=self.negative_slope)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    resx[smtidx][ind+1] = unbatch(x, dpt.batch)
        
        # make some histograms
        # h = torch.empty((data[0].num_graphs, len(self.scales)*len(self.tomobins), self.nbins, x.shape[1])).to(data[0].x.device)
        h = torch.empty((data[0].num_graphs, len(self.scales)*len(self.tomobins), self.nbins, len(self.convs_list[0])+1)).to(data[0].x.device)
        for sm in range(len(self.scales)):
            for t in range(len(self.tomobins)):
                smtidx = self.get_sm_t_idx(sm, t)
                for bi, bpt in enumerate(data[smtidx].to_data_list()):
                    for convidx in range(len(self.convs_list[0]) + 1):
                        h[bi, smtidx, :, convidx] = torchist.histogram(resx[smtidx][convidx][bi], edges=self.bins[smtidx])
                    # for nodeidx in range(dpt.x.shape[1]):
                    #     h[di, smtidx, :, nodeidx] = torchist.histogram(dpt.x[:, nodeidx], edges=self.bins[smtidx])
        h = h.reshape((data[0].num_graphs, self.hinp))
        
        x = h
        # x = self.input_norm(h)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x

def set_up_model(dataset, num_classes, device, num_subgraphs, loss_type="L1Loss"):
    scales, tomobins = np.load(dataset.scale), np.load(dataset.tomobin)
    num_classes, criterion = base_set_up_model(dataset, num_classes, device, loss_type)
    model = HistGINE(dataset.num_features, dataset.num_edge_features, num_classes, num_subgraphs, scales, tomobins, \
                        num_layers=2, dropout=0.2, negative_slope=0.2, \
                        internal_nn_layers=None, internal_nn_hidden_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) # changed lr
    return model, optimizer, criterion

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
from torch.nn import Linear, BatchNorm1d, ReLU, Parameter, Module
from torch_geometric.nn import aggr, MessagePassing
from torch_geometric.utils import unbatch, add_self_loops
import tqdm, gc, numpy as np
from BaseGNN import baseGNN, base_set_up_model, train, test, predict
from torch_geometric.nn.inits import ones

import matplotlib.pyplot as plt

class EdgeNNLayer(MessagePassing):
    def __init__(self, d0=45/60): # linking length in degrees
        super().__init__(aggr='mean') # get sum of contributions from neighbouring peaks
        self.d0 = d0
        self.beta = Parameter(torch.empty(1))
        self.reset_parameters()
        self.weight_aggr = aggr.SumAggregation()
    
    def reset_parameters(self):
        print("resetting parameters")
        # set to 1 / d0
        self.beta.data[0] = 1 / self.d0
        # ones(self.beta)
    
    def message(self, x_i, x_j, edge_attr):
        beta = self.beta.clone()
        return x_j * torch.exp(-beta * edge_attr[:, 0:1]), torch.exp(-beta * edge_attr[:, 0:1])
        # return x_i * self.beta, x_i * self.beta
    
    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0, num_nodes=x.shape[0])
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # add in the self contribution and then average over all nodes
        
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        vals, weights = inputs
        return self.aggr_module(vals, index, ptr=ptr, dim_size=dim_size, dim=self.node_dim)# / self.weight_aggr(weights, index, ptr, dim_size)

class EdgeNN(baseGNN):
    def __init__(self, node_features, edge_features, num_classes, num_subgraphs, scales, tomobins, \
                num_layers=3, dropout=0, negative_slope=0.2, nbins=14, \
                internal_nn_layers=2, internal_nn_hidden_channels=64):
        super(EdgeNN, self).__init__(node_features, edge_features, num_classes, 
                    num_subgraphs*num_layers*internal_nn_hidden_channels, num_layers, dropout, negative_slope)
        self.scales = scales
        self.tomobins = tomobins
        self.nbins = 14
        self.compute_bin_extrema()
        
        self.edge_layer = torch.nn.ModuleList([EdgeNNLayer()])
        
        # self.hinp = hinp = nbins*len(scales)*len(tomobins)*inhc
        self.hinp = hinp = nbins*len(scales)*len(tomobins) * 2  #*(num_layers+1)
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
        edgex = {}
        for sm in range(len(self.scales)):
            for t in range(len(self.tomobins)):
                smtidx = self.get_sm_t_idx(sm, t)
                dpt = data[smtidx]
                x, edge_index, edge_attr = dpt.x, dpt.edge_index.to(torch.int64), dpt.edge_attr
                resx[smtidx] = unbatch(x, dpt.batch)
                out = self.edge_layer[0](x, edge_index, edge_attr)
                edgex[smtidx] = unbatch(out, dpt.batch)
        
        # plt.figure(figsize=(20, 10))
        # kwargs1, kwargs2 = {"label": "og"}, {"label": "con"}
        # for i in range(8):
        #     plt.plot(np.arange(i*14, (i+1)*14), torchist.histogram(resx[i][0], edges=self.bins[0]).cpu().numpy(), color='b', **kwargs1)
        #     plt.plot(np.arange(i*14, (i+1)*14), torchist.histogram(edgex[i][0], edges=self.bins[0]).cpu().numpy(), color='r', **kwargs2)
        #     kwargs1, kwargs2 = {}, {}
        # plt.grid()
        # plt.legend()
        # plt.savefig("/global/cfs/cdirs/des/shubh/graphs/graphs_weak_lensing/EdgeNN.png")
        # plt.close()
        # print("saved")
        # exit()
        
        # make some histograms
        h = torch.empty((data[0].num_graphs, len(self.scales)*len(self.tomobins), self.nbins * 2)).to(data[0].x.device)
        for sm in range(len(self.scales)):
            for t in range(len(self.tomobins)):
                smtidx = self.get_sm_t_idx(sm, t)
                for bi, bpt in enumerate(edgex[smtidx]):
                    h[bi, smtidx, :self.nbins] = torchist.histogram(bpt, edges=self.bins[smtidx])
                for bi, bpt in enumerate(resx[smtidx]):
                    h[bi, smtidx, self.nbins:] = torchist.histogram(bpt, edges=self.bins[smtidx])
        h = h.reshape((data[0].num_graphs, self.hinp))
        
        x = h
        x = self.input_norm(h)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x

def set_up_model(dataset, num_classes, device, num_subgraphs, loss_type="L1Loss"):
    scales, tomobins = np.load(dataset.scale), np.load(dataset.tomobin)
    num_classes, criterion = base_set_up_model(dataset, num_classes, device, loss_type)
    model = EdgeNN(dataset.num_features, dataset.num_edge_features, num_classes, num_subgraphs, scales, tomobins, \
                        num_layers=2, dropout=0.2, negative_slope=0.2, \
                        internal_nn_layers=None, internal_nn_hidden_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # changed lr
    return model, optimizer, criterion

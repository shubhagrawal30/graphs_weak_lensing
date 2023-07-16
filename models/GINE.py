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
import tqdm, gc, numpy as np

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
    

# training, prediction, and evaluation code

def set_up_model(dataset, num_classes, device):
    print("initializing model")
    model = GINE(dataset.num_features, dataset.num_edge_features, num_classes, \
                    num_layers=4, dropout=0, negative_slope=0.2, \
                    internal_nn_layers=2, internal_nn_hidden_channels=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    return model, optimizer, criterion

def train(loader, model, optimizer, criterion, scaler, indices, device):
    model.train()
    loss_all = 0
    for data in tqdm.tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).cpu()
        true = data.y.reshape(-1, 6)[:, indices].cpu()
        torch.cuda.empty_cache()
        true = torch.tensor(scaler.transform(true), dtype=torch.float)
        loss = criterion(out, true)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        del data, out, true, loss
        gc.collect()
        torch.cuda.empty_cache()
    return loss_all / len(loader.dataset)

def test(loader, model, optimizer, criterion, scaler, indices, device):
    model.eval()
    with torch.no_grad():
        error = 0
        for data in tqdm.tqdm(loader):
            data = data.to(device)
            pred = model(data).cpu()
            true = data.y.reshape(-1, 6)[:, indices].cpu()
            true = torch.tensor(scaler.transform(true), dtype=torch.float)
            error += torch.sum((pred - true) ** 2)
            del data, pred, true
            gc.collect()
            torch.cuda.empty_cache()
    return error / len(loader.dataset)

def predict(loader, model, optimizer, criterion, scaler, indices, device):
    model.eval()
    out, true = np.array([]), np.array([])
    for data in tqdm.tqdm(loader):
        with torch.no_grad():
            data = data.to(device)
            out = np.append(out, scaler.inverse_transform(model(data).cpu().numpy()))
            true = np.append(true, data.y.cpu().numpy())
            del data
            gc.collect()
            torch.cuda.empty_cache()
    return out.reshape(-1, 2), true.reshape(-1, 6)[:, indices]



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

class GATv2(torch.nn.Module):
    def __init__(self, node_features, edge_features, num_classes, \
                 hidden_channels=8, num_layers=3, heads=8, dropout=0, negative_slope=0.2):
        super(GATv2, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.node_norm = BatchNorm(node_features)
        self.edge_norm = BatchNorm(edge_features)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(node_features, hidden_channels, \
                            edge_dim=edge_features, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, \
                            edge_dim=edge_features, heads=heads, dropout=dropout))
            
        self.readout = aggr.SoftmaxAggregation(learn=True)
        self.linear = Linear(hidden_channels * heads, num_classes)

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
    model = GATv2(dataset.num_features, dataset.num_edge_features, num_classes, \
                    hidden_channels=8, num_layers=4, heads=8, dropout=0, \
                    negative_slope=0.2).to(device)

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



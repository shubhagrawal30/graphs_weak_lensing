# base class for GNNs in this repository
# 
# note that our graphs have 1 node feature (the pixel value)
# and 2 edge features (the separation and angle between the two pixels)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from torch.nn import Linear
from torch_geometric.nn import aggr
import tqdm, gc, numpy as np

pred_type = None

class baseGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, num_classes, lin_in, \
                 num_layers=3, dropout=0, negative_slope=0.2):
        super(baseGNN, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.negative_slope = negative_slope

        # self.node_norm = BatchNorm(node_features)
        # self.edge_norm = BatchNorm(edge_features)
        
        # self.convs = torch.nn.ModuleList()
        
        # self.readout = aggr.SumAggregation()
        # self.readout = aggr.SoftmaxAggregation(learn=True)
        # self.linears = torch.nn.ModuleList([Linear(lin_in, lin_in), Linear(lin_in, lin_in)])
        # added a layer here
        # self.linear_fin = Linear(lin_in, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index.to(torch.int64), data.edge_attr
        x, edge_attr = self.node_norm(x), self.edge_norm(edge_attr)
        resx = {}
        for ind, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
            resx[ind] = torch.clone(x)
            # x = F.leaky_relu(x, negative_slope=self.negative_slope)
            # x = F.dropout(x, p=self.dropout, training=self.training)
            resx[ind] = self.readout(resx[ind], data.batch)
        # x = self.readout(x, data.batch)
        # for linear in self.linears:
        #     x = F.leaky_relu(linear(x), negative_slope=self.negative_slope)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.cat([resx[i] for i in range(len(self.convs))], dim=1)
        for linear in self.linears:
            h = F.leaky_relu(linear(h), negative_slope=self.negative_slope)
            h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.linear_fin(h)
        return x

# training, prediction, and evaluation code

def neg_log_likelihood(preds, y):
    # assuming first half of predictions are means and second half are log variances
    means, log_vars = preds[:, :preds.shape[1]//2], preds[:, preds.shape[1]//2:]
    error = y - means
    return torch.mean(0.5 * torch.exp(-log_vars) * error * error + 0.5 * log_vars)

def move_to_device(data, device):
    try:
        return data.to(device)
    except:
        return [move_to_device(d, device) for d in data]

def get_y(data):
    try:
        return data.y
    except:
        return get_y(data[0])
    
def get_num_graphs(data):
    try:
        return data.num_graphs
    except:
        return get_num_graphs(data[0])

def train(loader, model, optimizer, criterion, scaler, indices, device):
    model.train()
    loss_all = 0.0
    for data in tqdm.tqdm(loader):
        data = move_to_device(data, device)
        optimizer.zero_grad()
        out = model(data).cpu()
        true = get_y(data).reshape(-1, 6)[:, indices].cpu()
        torch.cuda.empty_cache()
        true = torch.tensor(scaler.transform(true), dtype=torch.float)
        loss = criterion(out, true)
        loss.backward()
        loss_all += get_num_graphs(data) * loss.item()
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
            data = move_to_device(data, device)
            pred = model(data).cpu()
            true = get_y(data).reshape(-1, 6)[:, indices].cpu()
            true = torch.tensor(scaler.transform(true), dtype=torch.float)
            loss = criterion(pred, true)
            error += get_num_graphs(data) * loss.item()
            del data, pred, true, loss
            gc.collect()
            torch.cuda.empty_cache()
    return error / len(loader.dataset)

def predict_for_mc_dropout(loader, model, optimizer, criterion, scaler, indices, device):
    model.train()
    n_pred = 10
    true = np.array([])
    preds = np.empty((10, len(loader.dataset), len(indices)*2))
    for i in range(n_pred):
        index = 0
        for data in tqdm.tqdm(loader):
            data = move_to_device(data, device)
            pred = model(data).cpu().detach().numpy()
            if i == 0:
                true = np.append(true, get_y(data).cpu())
            preds[i, index:index+get_num_graphs(data)] = pred
            index += get_num_graphs(data)
            del data, pred
            gc.collect()
            torch.cuda.empty_cache()
    
    return preds, true.reshape(-1, 6)[:, indices]

def predict_for_mse_loss(loader, model, optimizer, criterion, scaler, indices, device):
    model.eval()
    true, pred = np.array([]), np.empty((0, len(indices)))
    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            data = move_to_device(data, device)
            pred = np.append(pred, scaler.inverse_transform(model(data).cpu().numpy()), axis=0)
            true = np.append(true, get_y(data).cpu())
            del data
            gc.collect()
            torch.cuda.empty_cache()
    return pred, true.reshape(-1, 6)[:, indices]

def predict(loader, model, optimizer, criterion, scaler, indices, device):
    if pred_type == "mse" or pred_type == "L1Loss":
        return predict_for_mse_loss(loader, model, optimizer, criterion, scaler, indices, device)
    elif pred_type == "nll":
        return predict_for_mc_dropout(loader, model, optimizer, criterion, scaler, indices, device)
    else:
        raise ValueError("loss_type must be either mse or nll")

def base_set_up_model(dataset, num_classes, device, loss_type="L1Loss"):
    global pred_type
    print("initializing model")
    if loss_type == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_type == "nll":
        criterion = neg_log_likelihood
        print("first half of predictions are means and second half are log variances")
        num_classes *= 2
    elif loss_type == "L1Loss":
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError("loss_type must be either mse, L1Loss, or nll")
    pred_type = loss_type
    return num_classes, criterion
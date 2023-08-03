# a neural net for infering cosmology from just the peak counts
#
# used as a baseline for comparison with the GNN
#
# this model takes the graph data as input, extracts just the node features
# makes a 1d dataset by binning them, to get number counts vs peak values 
# and runs a non-Graph-based nn on it

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
import tqdm, gc, numpy as np

pred_type = None

class primitiveNN(torch.nn.Module):
    def __init__(self, input_features, num_classes, dropout=0, negative_slope=0.2,
                 num_dense_layers=4, dense_layer_size=128):
        super(primitiveNN, self).__init__()
        self.input_features = input_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.layers = torch.nn.ModuleList()
        self.layers.append(Linear(input_features, dense_layer_size))
        for i in range(num_dense_layers - 2):
            self.layers.append(Linear(dense_layer_size, dense_layer_size))
        self.layers.append(Linear(dense_layer_size, num_classes))
        self.input_norm = BatchNorm1d(self.input_features)

    def forward(self, data):
        x = self.input_norm(data)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x

def set_up_model(dataset, num_classes, device, loss_type="L1Loss"):
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

    model = primitiveNN(dataset.num_features, num_classes, num_dense_layers=3, \
                    dropout=0.1, negative_slope=0.2, dense_layer_size=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer, criterion

def neg_log_likelihood(preds, y):
    # assuming first half of predictions are means and second half are log variances
    means, log_vars = preds[:, :preds.shape[1]//2], preds[:, preds.shape[1]//2:]
    error = y - means
    return torch.mean(0.5 * torch.exp(-log_vars) * error * error + 0.5 * log_vars)

def train(loader, model, optimizer, criterion, scaler, indices, device):
    model.train()
    loss_all = 0.0
    for data, label in tqdm.tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).cpu()
        true = label.reshape(-1, 6)[:, indices].cpu()
        torch.cuda.empty_cache()
        true = torch.tensor(scaler.transform(true), dtype=torch.float)
        loss = criterion(out, true)
        loss.backward()
        loss_all += len(data) * loss.item()
        optimizer.step()
        del data, out, true, loss
        gc.collect()
        torch.cuda.empty_cache()
    return loss_all / len(loader.dataset)

def test(loader, model, optimizer, criterion, scaler, indices, device):
    model.eval()
    with torch.no_grad():
        error = 0
        for data, label in tqdm.tqdm(loader):
            data = data.to(device)
            pred = model(data).cpu()
            true = label.reshape(-1, 6)[:, indices].cpu()
            true = torch.tensor(scaler.transform(true), dtype=torch.float)
            loss = criterion(pred, true)
            error += len(data) * loss.item()
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
        for data, label in tqdm.tqdm(loader):
            data = data.to(device)
            pred = model(data).cpu().detach().numpy()
            if i == 0:
                true = np.append(true, label.cpu())
            preds[i, index:index+data.num_graphs] = pred
            index += data.num_graphs
            del data, pred
            gc.collect()
            torch.cuda.empty_cache()
    
    return preds, true.reshape(-1, 6)[:, indices]

def predict_for_mse_loss(loader, model, optimizer, criterion, scaler, indices, device):
    model.eval()
    true, pred = np.array([]), np.empty((0, len(indices)))
    with torch.no_grad():
        for data, label in tqdm.tqdm(loader):
            data = data.to(device)
            pred = np.append(pred, scaler.inverse_transform(model(data).cpu().numpy()), axis=0)
            true = np.append(true, label.cpu())
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

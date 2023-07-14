print("importing GATv2")
import sys
sys.path.append("../models/")
from GATv2 import GATv2

print("importing Patches")
from peaks_pygdata import Patches

print("importing dependencies")
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tqdm, gc
import matplotlib.pyplot as plt

print("importing torch")
import torch
from torch_geometric.loader import DataLoader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

num_epochs = 5
out_name = "20230714_GATv2"

print("loading dataset")
dataset_name = "20231107_patches_flatsky_fwhm3_radius8_noiseless"
dataset = Patches(dataset_name)
orig_labels = ["H0", "Ob", "Om", "ns", "s8", "w0"]
indices = orig_labels.index("Om"), orig_labels.index("s8")
num_classes = len(indices)

train_dataset = dataset[:256]
val_dataset = dataset[-128:]
batch_size = 16
print(batch_size, len(train_dataset) / batch_size, \
      len(train_dataset), len(val_dataset), len(dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("scaling")
scaler = MinMaxScaler()
true = np.array([])
for i, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
    true = np.append(true, data.y)

scaler.fit(true.reshape(-1, 6)[:, indices])

print("initializing model")
model = GATv2(dataset.num_features, dataset.num_edge_features, num_classes, \
                hidden_channels=64, num_layers=3, heads=8, dropout=0.1, \
                negative_slope=0.2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train(loader):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).cpu()
        true = data.y.reshape(-1, 6)[:, indices].cpu()
        loss = criterion(out, torch.tensor(scaler.transform(true)))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        del data, out, true, loss
        gc.collect()
    return loss_all / len(loader.dataset)

def test(loader):
    model.eval()
    error = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).cpu()
        true = data.y.reshape(-1, 6)[:, indices].cpu()
        error += np.sum((pred - torch.tensor(scaler.transform(true))) ** 2)
        del data, pred, true
        gc.collect()
    return error / len(loader.dataset)

for epoch in tqdm.tqdm(range(num_epochs)):
    print(f"Epoch {epoch}")
    train_loss = train(train_loader)
    val_loss = test(val_loader)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print("saving model")
torch.save(model.state_dict(), f"../outs/{out_name}/model.pt")

out, true = np.array([]), np.array([])
loader = train_loader

print("predicting")
model.eval()
for data in loader:
    data = data.to(device)
    out = np.append(out, scaler.inverse_transform(model(data).numpy()))
    true = np.append(true, data.y.numpy())
    del data
    gc.collect()

out = out.reshape(-1, 2)
true = true.reshape(-1, 6)[:, indices]

print("plotting")
fig, axs = plt.subplots(1, 2, figsize=(10, 8))
for i in range(2):
    axs[i].scatter(true[:, i], out[:, i], s=1)
    axs[i].set_xlabel("True")
    axs[i].set_ylabel("Predicted")
    axs[i].set_title(orig_labels[indices[i]])
    axs[i].plot([np.min(true[:, i]), np.max(true[:, i])], \
                [np.min(true[:, i]), np.max(true[:, i])], c="k")
    axs[i].grid()
plt.tight_layout()
plt.savefig(f"../outs/{out_name}/pred-true.png")
plt.close()

fig, axs = plt.subplots(1, 2, figsize=(10, 8))
for i in range(2):
    axs[i].hist(true[:, i] - out[:, i], bins=100)
    axs[i].set_xlabel("True - Predicted")
    axs[i].set_ylabel("Count")
    axs[i].set_title(orig_labels[indices[i]])
    axs[i].grid()
plt.tight_layout()
plt.savefig(f"../outs/{out_name}/hist.png")
plt.close()
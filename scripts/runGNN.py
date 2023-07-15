print("importing GATv2")
import sys, pathlib, time
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

num_epochs = 25
out_name = "20230714_GATv2"
pathlib.Path(f"../outs/{out_name}").mkdir(parents=True, exist_ok=True)

print("loading dataset")
dataset_name = "20231107_patches_flatsky_fwhm3_radius8_noiseless"
dataset = Patches(dataset_name)
orig_labels = ["H0", "Ob", "Om", "ns", "s8", "w0"]
indices = orig_labels.index("Om"), orig_labels.index("s8")
num_classes = len(indices)

batch_size = 96
train_dataset, val_dataset, test_dataset = dataset[:int(0.8 * len(dataset))], \
    dataset[int(0.8 * len(dataset)):int(0.9 * len(dataset))], dataset[int(0.9 * len(dataset)):]
print(batch_size, len(train_dataset) / batch_size, \
      len(train_dataset), len(val_dataset), len(test_dataset), len(dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("scaling")
scaler = MinMaxScaler()
true = np.array([])
for i, data in enumerate(train_loader):
    true = np.append(true, data.y)

scaler.fit(true.reshape(-1, 6)[:, indices])

print("initializing model")
model = GATv2(dataset.num_features, dataset.num_edge_features, num_classes, \
                hidden_channels=8, num_layers=4, heads=8, dropout=0, \
                negative_slope=0.2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train(loader):
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

def test(loader):
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

def predict(loader):
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

# save all training and validation losses and the model with best validation loss
best_val_loss = np.inf
best_epoch = -1
print("training")
with open(f"../outs/{out_name}/log.txt", "w") as f:
    for epoch in range(num_epochs):
        start = time.time()
        print(f"Epoch {epoch}")
        train_loss = train(train_loader)
        val_loss = test(val_loader)

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        f.write(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')
        print(f"Time: {time.time() - start:.4f}s")
        f.write(f"Time: {time.time() - start:.4f}s\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"../outs/{out_name}/best_model.pt")

        del train_loss, val_loss
        gc.collect()
        torch.cuda.empty_cache()

print("saving model")
torch.save(model.state_dict(), f"../outs/{out_name}/last_model.pt")

def plotting(model, pred_true_filename, hist_filenames):
    print("predicting")
    train_out, train_true = predict(train_loader)
    val_out, val_true = predict(val_loader)
    test_out, test_true = predict(test_loader)

    print("plotting")
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    for i in range(2):
        axs[i].scatter(train_true[:, i], train_out[:, i], s=1, label="Train")
        axs[i].scatter(val_true[:, i], val_out[:, i], s=1, label="Validation")
        axs[i].scatter(test_true[:, i], test_out[:, i], s=1, label="Test")
        axs[i].set_xlabel("True")
        axs[i].set_ylabel("Predicted")
        axs[i].set_title(orig_labels[indices[i]])
        axs[i].plot([np.min(train_true[:, i]), np.max(train_true[:, i])], \
                    [np.min(train_true[:, i]), np.max(train_true[:, i])], c="k")
        axs[i].legend()
        axs[i].grid()
    plt.tight_layout()
    plt.savefig(pred_true_filename)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    for i in range(2):
        axs[i].hist(train_true[:, i] - train_out[:, i], bins=100, label="Train", histtype="step", density=True)
        axs[i].hist(val_true[:, i] - val_out[:, i], bins=100, label="Validation", histtype="step", density=True)
        axs[i].hist(test_true[:, i] - test_out[:, i], bins=100, label="Test", histtype="step", density=True)
        axs[i].set_xlabel("True - Predicted")
        axs[i].set_ylabel("Count")
        axs[i].set_title(orig_labels[indices[i]])
        axs[i].legend()
        axs[i].grid()
    plt.tight_layout()
    plt.savefig(hist_filenames)
    plt.close()

plotting(model, f"../outs/{out_name}/pred_true.png", f"../outs/{out_name}/hist.png")

if best_epoch != num_epochs - 1:
    print(f"Loading best model from epoch {best_epoch}")
    model.load_state_dict(torch.load(f"../outs/{out_name}/best_model.pt"))
    plotting(model, f"../outs/{out_name}/best_pred_true.png", f"../outs/{out_name}/best_hist.png")

print("done!")
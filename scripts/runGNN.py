print("importing GATv2")
import sys, pathlib, time
sys.path.append("../models/")

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

# from GATv2 import GATv2, set_up_model, train, test, predict
# out_name = "20230714_GATv2"

from GINE import GINE, set_up_model, train, test, predict
out_name = "20230714_GINE"

num_epochs = 5
pathlib.Path(f"../outs/{out_name}/chkpts/").mkdir(parents=True, exist_ok=True)

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
model, optimizer, criterion = set_up_model(dataset, num_classes, device)
args = model, optimizer, criterion, scaler, indices, device

# save all training and validation losses and the model with best validation loss
best_val_loss = np.inf
best_epoch = -1
print("training")
for epoch in range(num_epochs):
    with open(f"../outs/{out_name}/log.txt", "a") as f:
        start = time.time()
        print(f"Epoch {epoch}")
        train_loss = train(train_loader, *args)
        val_loss = test(val_loader, *args)

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        f.write(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')
        print(f"Time: {time.time() - start:.4f}s")
        f.write(f"Time: {time.time() - start:.4f}s\n")

        torch.save(model.state_dict(), f"../outs/{out_name}/chkpts/{epoch}.pt")

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
    train_out, train_true = predict(train_loader, *args)
    val_out, val_true = predict(val_loader, *args)
    test_out, test_true = predict(test_loader, *args)

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
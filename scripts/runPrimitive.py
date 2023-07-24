import sys, pathlib, time, os
sys.path.append("../models/")

print("importing data modules")
from peaks_pygdata import Patches
from peaks_primitive import Histograms

print("importing dependencies")
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tqdm, gc, datetime
import matplotlib.pyplot as plt
import joblib 

print("importing torch")
import torch
from torch.utils.data import DataLoader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

from PrimitiveNN import primitiveNN, set_up_model, train, test, predict
out_name = "20230720_past"

num_epochs = 100
pathlib.Path(f"../outs/{out_name}/chkpts/").mkdir(parents=True, exist_ok=True)
overwrite_epochs = False
overwrite_logs = False

if overwrite_logs:
    if os.path.exists(f"../outs/{out_name}/log.txt"):
        os.remove(f"../outs/{out_name}/log.txt")
else:
    with open(f"../outs/{out_name}/log.txt", "a") as f:
        f.write("Starting new run\n at " + str(datetime.datetime.now()) + "\n")

print("loading dataset")
# dataset_name = "20231107_patches_flatsky_fwhm3_radius8_noiseless"
orig_labels = ["H0", "Ob", "Om", "ns", "s8", "w0"]
indices = orig_labels.index("Om"), orig_labels.index("s8")
num_classes = len(indices)

dataset = Histograms()
batch_size = 96
train_dataset, val_dataset, test_dataset = dataset[:int(0.8 * len(dataset))], \
    dataset[int(0.8 * len(dataset)):int(0.9 * len(dataset))], dataset[int(0.9 * len(dataset)):]
# train_dataset, val_dataset, test_dataset = dataset[:batch_size*], \
#     dataset[batch_size*10:batch_size*11], dataset[batch_size*11:batch_size*12]

print(batch_size, len(train_dataset) / batch_size, \
        len(train_dataset), len(val_dataset), len(test_dataset), len(dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("scaling")
try:
    scaler = joblib.load(f"../outs/{out_name}/scaler.pkl")
except:
    scaler = MinMaxScaler()
    true = np.array([])
    for i, (data, label) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        true = np.append(true, label)
    scaler.fit(true.reshape(-1, 6)[:, indices])
    joblib.dump(scaler, f"../outs/{out_name}/scaler.pkl")

model, optimizer, criterion = set_up_model(dataset, num_classes, device)
args = model, optimizer, criterion, scaler, indices, device

# save all training and validation losses and the model with best validation loss
best_val_loss = np.inf
best_epoch = -1
print("training")
for epoch in range(num_epochs):
    with open(f"../outs/{out_name}/log.txt", "a") as f:
        if not overwrite_epochs and os.path.exists(f"../outs/{out_name}/chkpts/{epoch}.pt"):
            model.load_state_dict(torch.load(f"../outs/{out_name}/chkpts/{epoch}.pt"))
            print(f"Loaded model from epoch {epoch}")
            f.write(f"Loaded model from epoch {epoch}\n")
            continue
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
            f.write(f"New best model saved at epoch {epoch}\n")

        del train_loss, val_loss
        gc.collect()
        torch.cuda.empty_cache()

print("saving model")
torch.save(model.state_dict(), f"../outs/{out_name}/last_model.pt")

def plotting_for_mc_dropout(loader, pred_true_filename, hist_filenames):
    print("predicting")
    preds, true = predict(loader, *args)

    np.save(f"../outs/{out_name}/preds.npy", preds)
    np.save(f"../outs/{out_name}/true.npy", true)

    preds_best, preds_std = preds[:,:, :preds.shape[-1]//2], preds[:,:, preds.shape[-1]//2:]

    predictions = np.empty((preds_best.shape[0] * 100, preds_best.shape[1], preds_best.shape[2]))
    for i in range(preds_best.shape[0]):
        for j in range(100):
            predictions[i*100+j] = np.random.normal(preds_best[i], np.exp(preds_std[i]))

    # inverse transform the predictions
    predictions = scaler.inverse_transform(predictions.reshape(-1, 2)).reshape(predictions.shape)

    predictions_best = np.nanmean(predictions, axis=0)
    predictions_std = np.nanstd(predictions, axis=0)

    print("plotting")
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    for i in range(2):
        axs[i].errorbar(true[:, i], predictions_best[:, i], yerr=predictions_std[:, i], \
                        marker="x", ls='none', alpha=0.4)
        axs[i].set_xlabel("True")
        axs[i].set_ylabel("Predicted")
        axs[i].set_title(orig_labels[indices[i]])
        axs[i].plot([np.min(true[:, i]), np.max(true[:, i])], \
                    [np.min(true[:, i]), np.max(true[:, i])], c="k")
        axs[i].grid()
    plt.tight_layout()
    plt.savefig(pred_true_filename)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    for i in range(2):
        axs[i].hist((true[:, i] - predictions_best[:, i]) / predictions_std[:, i], bins=50)
        axs[i].set_xlabel("True - Predicted / Uncertainty")
        axs[i].set_ylabel("Count")
        axs[i].set_title(orig_labels[indices[i]])
        axs[i].grid()
    plt.tight_layout()
    plt.savefig(hist_filenames)
    plt.close()

def plotting_for_mse_loss(loader, pred_true_filename, hist_filenames):
    print("predicting")
    preds, true = predict(loader, *args)

    np.save(f"../outs/{out_name}/preds.npy", preds)
    np.save(f"../outs/{out_name}/true.npy", true)

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    for i in range(2):
        axs[i].scatter(true[:, i], preds[:, i], s=1)
        axs[i].set_xlabel("True")
        axs[i].set_ylabel("Predicted")
        axs[i].set_title(orig_labels[indices[i]])
        axs[i].plot([np.min(true[:, i]), np.max(true[:, i])], \
                    [np.min(true[:, i]), np.max(true[:, i])], c="k")
        axs[i].grid()
    plt.tight_layout()
    plt.savefig(pred_true_filename)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    for i in range(2):
        axs[i].hist(true[:, i] - preds[:, i], bins=100, histtype="step", density=True)
        axs[i].set_xlabel("True - Predicted")
        axs[i].set_ylabel("Count")
        axs[i].set_title(orig_labels[indices[i]])
        axs[i].grid()
    plt.tight_layout()
    plt.savefig(hist_filenames)
    plt.close()


plotting = plotting_for_mse_loss
plotting(train_loader, f"../outs/{out_name}/pred_true.png", f"../outs/{out_name}/hist.png")

if best_epoch != num_epochs - 1:
    print(f"Loading best model from epoch {best_epoch}")
    model.load_state_dict(torch.load(f"../outs/{out_name}/best_model.pt"))
    plotting(train_loader, f"../outs/{out_name}/best_pred_true.png", f"../outs/{out_name}/best_hist.png")
else:
    print("best model is last model")
    with open(f"../outs/{out_name}/log.txt", "a") as f:
        f.write("best model is last model\n")

print("done!")
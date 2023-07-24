import sys, pathlib, time, os
sys.path.append("../models/")

print("importing Patches")
from peaks_pygdata import Patches

print("importing dependencies")
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tqdm, gc
import matplotlib.pyplot as plt
import joblib 

print("importing torch")
import torch
from torch_geometric.loader import DataLoader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

from GATv2 import GATv2, set_up_model, train, test, predict
out_name = "20230716_GATv2"
checkpoint = f"../outs/{out_name}/best_model.pt"
pathlib.Path(f"../outs/{out_name}/chkpts/").mkdir(parents=True, exist_ok=True)

# from GINE import GINE, set_up_model, train, test, predict
# out_name = "20230714_GINE"

print("loading dataset")
dataset_name = "20231107_patches_flatsky_fwhm3_radius8_noiseless"
dataset = Patches(dataset_name)
orig_labels = ["H0", "Ob", "Om", "ns", "s8", "w0"]
indices = orig_labels.index("Om"), orig_labels.index("s8")
num_classes = len(indices)

batch_size = 96
train_dataset, val_dataset, test_dataset = dataset[:int(0.8 * len(dataset))], \
    dataset[int(0.8 * len(dataset)):int(0.9 * len(dataset))], dataset[int(0.9 * len(dataset)):]
# train_dataset, val_dataset, test_dataset = dataset[:batch_size*5], \
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
    for i, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        true = np.append(true, data.y)
    scaler.fit(true.reshape(-1, 6)[:, indices])
    joblib.dump(scaler, f"../outs/{out_name}/scaler.pkl")

model, optimizer, criterion = set_up_model(dataset, num_classes, device)
args = model, optimizer, criterion, scaler, indices, device

# load a model from a checkpoint
model.load_state_dict(torch.load(checkpoint))

def plotting(pred_true_filename, hist_filenames):
    print("predicting")
    preds, true = predict(test_loader, *args)

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
        axs[i].set_xlabel("True - Predicted")
        axs[i].set_ylabel("Count")
        axs[i].set_title(orig_labels[indices[i]])
        axs[i].grid()
    plt.tight_layout()
    plt.savefig(hist_filenames)
    plt.close()

plotting(f"../outs/{out_name}/rand_pred_true.png", f"../outs/{out_name}/rand_hist.png")

print("done!")
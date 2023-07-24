import torch
from torch.utils.data import Dataset
import os, tqdm, shutil, sys, pathlib
import numpy as np
from collections.abc import Iterable

if os.uname()[1].endswith("marmalade.physics.upenn.edu") or os.uname()[1][:4] == "node":
    print("I'm on marmalade!")
    SYSTEM_NAME = "marmalade"
    DATA_PATH = "/data2/shared/shubh/peaks/"
    pathlib.Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
elif os.uname()[1][:5] == "login" or os.uname()[1][:3] == "nid":
    print("I'm on perlmutter!")
    SYSTEM_NAME = "perlmutter"
    sys.exit("I don't know what to do on perlmutter!")
else:
    sys.exit("I don't know what computer I'm on!")

N_BINS = 25 # edge bins are +-np.inf
RANGE = (-3, 0) # logspace

class Histograms(Dataset):
    @property
    def labels_file(self) -> str:
        return f"labels_{self.nbins}_{self.range[0]}_{self.range[1]}.npy"
    
    @property
    def hist_file(self) -> str:
        return f"histograms_{self.nbins}_{self.range[0]}_{self.range[1]}.npy"
    
    @property
    def bins(self) -> np.ndarray:
        return np.concatenate(([-np.inf],
                np.logspace(*self.range, self.nbins-1), [np.inf]))
    
    def process(self):
        self.labels = np.empty((len(self.Patches_data), self.nlabels))
        self.histograms = np.empty((len(self.Patches_data), self.nbins))
        for i, data in tqdm.tqdm(enumerate(self.Patches_data), total=len(self.Patches_data)):
            self.labels[i] = data.y
            self.histograms[i] = np.histogram(data.x, bins=self.bins)[0]
    
    def __init__(self, Patches_data=None, nbins=N_BINS, range=RANGE, overwrite=False, \
                 labels=None, histograms=None):
        if overwrite:
            if os.path.exists(os.path.join(DATA_PATH, self.labels_file)):
                os.remove(os.path.join(DATA_PATH, self.labels_file))
            if os.path.exists(os.path.join(DATA_PATH, self.hist_file)):
                os.remove(os.path.join(DATA_PATH, self.hist_file))
        
        self.nbins = nbins
        self.range = range
        self.Patches_data = Patches_data
        self.num_features = self.nbins

        if labels is not None and histograms is not None:
            self.labels = labels
            self.histograms = histograms
            self.nlabels = len(self.labels[0])
            self.num_features = self.nbins    
            return
        
        self.labels = None
        self.histograms = None

        if self.Patches_data is not None:
            self.nlabels = len(self.Patches_data[0].y)
        
        if os.path.exists(os.path.join(DATA_PATH, self.labels_file)):
            print("Loading labels")
            self.labels = np.load(os.path.join(DATA_PATH, self.labels_file))
            if self.Patches_data is not None:
                assert len(self.labels) == len(self.Patches_data)
                assert len(self.labels[0]) == self.nlabels
            else:
                self.nlabels = len(self.labels[0])
        
        if os.path.exists(os.path.join(DATA_PATH, self.hist_file)):
            print("Loading histograms")
            self.histograms = np.load(os.path.join(DATA_PATH, self.hist_file))
            if self.Patches_data is not None:
                assert len(self.histograms) == len(self.Patches_data)
            else:
                assert len(self.histograms) == len(self.labels)
            assert len(self.histograms[0]) == self.nbins

        if self.labels is None or self.histograms is None:
            assert self.Patches_data is not None
            self.process()
            np.save(os.path.join(DATA_PATH, self.labels_file), self.labels)
            np.save(os.path.join(DATA_PATH, self.hist_file), self.histograms)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if type(idx) is int:
            return torch.Tensor(self.histograms[idx]), torch.Tensor(self.labels[idx])
        if type(idx) is slice:
            idx = np.arange(*idx.indices(len(self)))
        if isinstance(idx, Iterable):
            return Histograms(labels=self.labels[idx], histograms=self.histograms[idx])
    
    
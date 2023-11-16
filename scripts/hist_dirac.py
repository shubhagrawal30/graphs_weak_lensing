import torch
from torch.utils.data import Dataset
import os, tqdm, shutil, sys, pathlib, pickle
import numpy as np
from collections.abc import Iterable

if os.uname()[1].endswith("marmalade.physics.upenn.edu") or os.uname()[1][:4] == "node":
    print("I'm on marmalade!")
    PEAKS_PATH = None # you should be generating the dataset on perlmutter
    HISTS_PATH = lambda dataset_name: f'/home2/shubh/graphs_weak_lensing/data/{dataset_name}/hists/'
elif os.uname()[1][:5] == "login" or os.uname()[1][:3] == "nid":
    print("I'm on perlmutter!")
    SYSTEM_NAME = "perlmutter"
    PEAKS_PATH = "/pscratch/sd/s/shubh/graph_data/dirac/"
    HISTS_PATH = lambda dataset_name: f'/global/cfs/cdirs/des/shubh/graphs/graphs_weak_lensing/data/{dataset_name}/hists/'
else:
    sys.exit("I don't know what computer I'm on!")

LABELS = ['om', 'h', 's8', 'w', 'ob', 'ns']

class DiracHistograms(Dataset):
    @property
    def raw_dir(self) -> str:
        return PEAKS_PATH
    
    @property
    def labels_file(self) -> str:
        return os.path.join(HISTS_PATH(self.dataset_name), f"labels.npy")
    
    @property
    def hist_file(self) -> str:
        return os.path.join(HISTS_PATH(self.dataset_name), f"histograms.npy")
    
    @property
    def scale(self):
        return self.dataset_name.split("_scale")[1].split("_")[0]
    
    @property
    def tomobin(self):
        return int(self.dataset_name.split("_tomobin")[1].split("_")[0])
    
    def process(self):
        filenames = os.listdir(self.raw_dir)
        if "done" in filenames:
            filenames.remove("done")
        self.labels = np.empty((len(filenames), len(LABELS)))
        
        for i, filename in tqdm.tqdm(enumerate(filenames), total=len(filenames)):
            with open(os.path.join(self.raw_dir, filename), "rb") as f:
                mmap, cosmo = pickle.load(f)
                hists = np.array(mmap.peaks[self.field_name][self.tomobin][self.scale])
            self.labels[i] = np.array([cosmo[label] for label in LABELS])
            if self.histograms is None:
                self.histograms = np.empty((len(filenames), len(hists)))
            self.histograms[i] = hists
    
    def __init__(self, dataset_name, field_name='k_sm_kE', overwrite=False, labels=None, histograms=None):
        self.dataset_name = dataset_name
        self.field_name = field_name
        os.makedirs(HISTS_PATH(self.dataset_name), exist_ok=True)
        
        if overwrite:
            for filename in [self.labels_file, self.hist_file]:
                if os.path.exists(filename):
                    os.remove(filename)

        if labels is not None and histograms is not None:
            # for slicing purposes
            self.labels = labels
            self.histograms = histograms
            self.nlabels = len(self.labels[0])
            return
        
        self.labels = None
        self.histograms = None
        
        if os.path.exists(self.labels_file):
            print("loading labels")
            self.labels = np.load(self.labels_file)
            self.nlabels = len(self.labels[0])
        
        if os.path.exists(self.hist_file):
            print("loading histograms")
            self.histograms = np.load(self.hist_file)

        if self.labels is None or self.histograms is None:
            self.process()
            np.save(self.labels_file, self.labels)
            np.save(self.hist_file, self.histograms)
        
        assert len(self.histograms) == len(self.labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if type(idx) is int:
            return torch.Tensor(self.histograms[idx]), torch.Tensor(self.labels[idx])
        if type(idx) is slice:
            idx = np.arange(*idx.indices(len(self)))
        if isinstance(idx, Iterable):
            return DiracHistograms(labels=self.labels[idx], histograms=self.histograms[idx])
    
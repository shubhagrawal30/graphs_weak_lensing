import torch
from torch.utils.data import Dataset
import os, tqdm, shutil, sys, pathlib, pickle
import numpy as np
from collections.abc import Iterable
import multiprocessing as mp

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

NTHREADS = 128
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
        return os.path.join(HISTS_PATH(self.dataset_name), f"../scales.npy")
    
    @property
    def tomobin(self):
        return os.path.join(HISTS_PATH(self.dataset_name), f"../tomobins.npy")
    
    def write_scales_and_tomobins(self, scales, tomobins):
        np.save(self.scale, scales)
        np.save(self.tomobin, tomobins)
    
    def process_one_file(self, args):
        ind, filename = args
        print(f"processing {ind}:{filename}", flush=True)
        
        hists = np.array([])
        scales = np.load(self.scale)
        tomobins = np.load(self.tomobin)
        with open(os.path.join(self.raw_dir, filename), "rb") as f:
            mmap, cosmo = pickle.load(f)
            for tb in tomobins:
                for sc in scales:
                    hists = np.append(hists, mmap.peaks[self.field_name][tb][sc])
        print(f"done {ind}:{filename}", flush=True)
        return hists, np.array([cosmo[label] for label in LABELS])
    
    def process(self):
        filenames = os.listdir(self.raw_dir)
        if "done" in filenames:
            filenames.remove("done")
        self.labels = np.empty((len(filenames), len(LABELS)))
        
        hists, _ = self.process_one_file([0, filenames[0]])
        self.histograms = np.empty((len(filenames), len(hists)))
        self.num_features = len(hists)
        
        pool = mp.Pool(NTHREADS)
        results = pool.map(self.process_one_file, enumerate(filenames))
        pool.close()
        pool.join()
        
        for i, (hists, labels) in enumerate(results):
            self.histograms[i] = hists
            self.labels[i] = labels
    
    def __init__(self, dataset_name, scales=None, tomobins=None, field_name='k_sm_kE', overwrite=False, labels=None, histograms=None):
        self.dataset_name = dataset_name
        self.field_name = field_name
        os.makedirs(HISTS_PATH(self.dataset_name), exist_ok=True)
        
        if scales is None:
            assert os.path.exists(self.scale)
        if tomobins is None:
            assert os.path.exists(self.tomobin)
        
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
            self.num_features = len(self.histograms[0])

        if self.labels is None or self.histograms is None:
            self.write_scales_and_tomobins(scales, tomobins)
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
            return DiracHistograms(self.dataset_name, labels=self.labels[idx], histograms=self.histograms[idx])
    
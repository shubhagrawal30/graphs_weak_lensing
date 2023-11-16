import torch
from torch_geometric.data import Dataset, Data
import os, tqdm, shutil, sys, pickle, time, pathlib
import numpy as np
import healpy as hp
import multiprocessing as mp

nthreads = 128

try:
    os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "1"
    from mpi4py import MPI
    MPI_ON = True
    print(MPI.Get_library_version(), flush=True)
except:
    MPI_ON = False
    pass

if os.uname()[1].endswith("marmalade.physics.upenn.edu") or os.uname()[1][:4] == "node":
    print("I'm on marmalade!")
    SYSTEM_NAME = "marmalade"
    PEAKS_PATH = None # you should be generating the dataset on perlmutter
    GRAPHS_PATH = lambda dataset_name: f'/home2/shubh/graphs_weak_lensing/data/{dataset_name}/graphs'
elif os.uname()[1][:5] == "login" or os.uname()[1][:3] == "nid":
    print("I'm on perlmutter!")
    SYSTEM_NAME = "perlmutter"
    PEAKS_PATH = "/pscratch/sd/s/shubh/graph_data/dirac/"
    GRAPHS_PATH = lambda dataset_name: f'/global/cfs/cdirs/des/shubh/graphs/graphs_weak_lensing/data/{dataset_name}/graphs/'
else:
    sys.exit("I don't know what computer I'm on!")

NUM_BATCHES = 100
LABELS = ['om', 'h', 's8', 'w', 'ob', 'ns']

class DiracPatches(Dataset):
    @property
    def raw_dir(self) -> str:
        return PEAKS_PATH
    
    @property
    def processed_dir(self) -> str:
        return GRAPHS_PATH(self.dataset_name)

    @property
    def raw_file_names(self):
        return ["done"]

    @property
    def processed_file_names(self):
        return [f'data{ind}.pt' for ind in range(self.num_batches)]
    
    @property
    def scale(self):
        return self.dataset_name.split("_scale")[1].split("_")[0]
    
    @property
    def tomobin(self):
        return int(self.dataset_name.split("_tomobin")[1].split("_")[0])

    def len(self):
        if SYSTEM_NAME == "perlmutter":
            return len(os.listdir(self.raw_dir)) - 1
        elif SYSTEM_NAME == "marmalade":
            with open(os.path.join(self.processed_dir, "done"), "r") as f:
                return int(f.read())  

    def __init__(self, dataset_name, num_batches=NUM_BATCHES, field_name='k_sm_kE', radius=45, \
                overwrite=False, transform=None, pre_transform=None, pre_filter=None):
        # dataset name should have _tomobin***_scale*** in it
        self.dataset_name = dataset_name
        self.num_batches = num_batches
        self.field_name = field_name
        self.radius = radius / 60 / 180 * np.pi # input assumed in arcmins, converted to radians later
        
        pathlib.Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        
        if overwrite:
            shutil.rmtree(self.processed_dir)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        filenames = os.listdir(self.raw_dir)
        if "done" in filenames: 
            filenames.remove("done")
        self.filenames_batched = np.array_split(filenames, self.num_batches)
        self.indices_batched = np.array_split(np.arange(self.len()), self.num_batches)
    
        super().__init__(None, transform, pre_transform, pre_filter)

        self.loaded_batch_ind = None
        self.loaded_batch = None

    def process_one_file(self, filename):
        with open(os.path.join(self.raw_dir, filename), 'rb') as f:
            mmap, cosmo = pickle.load(f)
            nside = self.nside = mmap.conf['nside']
            peaks_hpinds = np.array(mmap.peaks[self.field_name][self.tomobin][self.scale+"_loc"])
            peaks_vals = np.array(mmap.peaks[self.field_name][self.tomobin][self.scale+"_val"])
            del mmap
        peaks_hpvec = np.vstack(hp.pix2vec(nside, peaks_hpinds))
        peak_key = lambda hpind: np.where(peaks_hpinds == hpind)[0][0]
        
        y = torch.tensor([cosmo[l] for l in LABELS], dtype=torch.float) # graph labels, shape: num_nodes, num_features
        x = torch.tensor(peaks_vals.reshape(-1, 1), dtype=torch.float) # node featuresnp.sta
        edge_index = torch.empty((2, 0), dtype=torch.int) # edges, in COO format, shape (2, num_edges)
        edge_attr = torch.empty((0, 2), dtype=torch.float) # edge features: separation, angle, shape (num_edges, num_features)
        
        # making edges
        for key in range(len(peaks_hpinds)):
            neigh_hpinds = hp.query_disc(nside, peaks_hpvec[:, key], self.radius, inclusive=True)
            apeaks_hpinds = np.intersect1d(peaks_hpinds, neigh_hpinds)
            apeaks_hpinds = apeaks_hpinds[apeaks_hpinds != peaks_hpinds[key]]
            apeaks_key = np.array([peak_key(hpind) for hpind in apeaks_hpinds]) # not using np.vectorize as apeaks is expected to be small
            
            edge_index = torch.cat((edge_index, torch.tensor(np.stack((np.repeat(key, len(apeaks_key)), apeaks_key)), dtype=torch.int)), dim=1)
            ra, dec = hp.pix2ang(nside, peaks_hpinds[key], lonlat=True)
            ara, adec = hp.pix2ang(nside, apeaks_hpinds, lonlat=True)
            seperations = np.sqrt((ara - ra)**2 + (adec - dec)**2) # flat sky approximation
            angles = np.arctan2(ara - ra, adec - dec)
            edge_attr = torch.cat((edge_attr, torch.tensor(np.stack((seperations, angles)).T, dtype=torch.float)))
        
        datapoint = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        return datapoint
        
    
    def process_one_batch(self, batch_ind):
        # randomize which batch we choose:
        start = time.time()
        batch_ind = (batch_ind + np.random.randint(self.num_batches)) % self.num_batches
        if os.path.exists(self.processed_paths[batch_ind]):
            print(f"Batch {batch_ind} already processed", flush=True)
            return
        print(f"Processing batch {batch_ind}", flush=True)
        filenames = self.filenames_batched[batch_ind]
        data_list = []
        
        pool = mp.Pool(nthreads)
        data_list = pool.map(self.process_one_file, filenames)
        pool.close()
        pool.join()
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        torch.save(data_list, self.processed_paths[batch_ind])
        print(f"Batch {batch_ind} processed in {time.time() - start} seconds", flush=True)
        return

    def process(self):
        if MPI_ON:
            run_count = 0
            comm = MPI.COMM_WORLD
            print(comm)
            while run_count < self.num_batches:
                if (run_count + comm.rank) < self.num_batches:
                    try:
                        self.process_one_batch(run_count + comm.rank)
                    except:
                        pass
                run_count += comm.size
        else:
            for batch_ind in range(self.num_batches):
                self.process_one_batch(batch_ind)

    def index_to_batch(self, idx):
        for batch_ind, indices in enumerate(self.indices_batched):
            if idx >= indices[0] and idx <= indices[-1]:
                return batch_ind
            
    def index_to_pos_in_batch(self, idx, batch_ind=None):
        if batch_ind is None:
            batch_ind = self.index_to_batch(idx)
        return idx - self.indices_batched[batch_ind][0]

    def get(self, idx):
        batch_ind = self.index_to_batch(idx)
        pos_in_batch = self.index_to_pos_in_batch(idx, batch_ind)
        if self.loaded_batch_ind is not None and self.loaded_batch_ind == batch_ind:
            return self.loaded_batch[pos_in_batch]
        else:
            self.loaded_batch_ind = batch_ind
            self.loaded_batch = torch.load(self.processed_paths[batch_ind])
            return self.loaded_batch[pos_in_batch]
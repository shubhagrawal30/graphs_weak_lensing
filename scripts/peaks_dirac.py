import torch
from torch_geometric.data import Dataset, Data
import os, tqdm, shutil, sys, pickle, time, pathlib, gc, psutil
import numpy as np
import healpy as hp
import multiprocessing as mp
from numba import jit

if __name__ == '__main__' or True:
    print(f'nthreads = {psutil.cpu_count(logical=True)}')
    print(f'ncores = {psutil.cpu_count(logical=False)}')
    print(f'nthreads_per_core = {psutil.cpu_count(logical=True) // psutil.cpu_count(logical=False)}')
    print(f'nthreads_available = {len(os.sched_getaffinity(0))}')
    print(f'ncores_available = {len(os.sched_getaffinity(0)) // (psutil.cpu_count(logical=True) // psutil.cpu_count(logical=False))}')

nthreads = 32

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

NUM_BATCHES = 1500 # get slightly higher than 32, so we can use 32 batch size (~1-2 in memory) or 64 (~2 in memory)
LABELS = ['om', 'h', 's8', 'w', 'ob', 'ns']

@jit(nopython=True)
def process_one_sc_tb_jit(sc, tb, all_peaks, all_neigh_hpinds, pix2ang, cosmo):
    # start = time.time()
    # print(filename, sc, tb, flush=True)
    edge_index = torch.empty((2, 0), dtype=torch.int) # edges, in COO format, shape (2, num_edges)
    edge_attr = torch.empty((0, 2), dtype=torch.float) # edge features: separation, angle, shape (num_edges, num_features)
    
    peaks_hpinds = np.array(all_peaks[tb][f"{sc:.1f}"+"_loc"])
    peaks_vals = np.array(all_peaks[tb][f"{sc:.1f}"+"_val"])
    
    peak_key = lambda hpind: np.where(peaks_hpinds == hpind)[0][0]
    
    y = torch.tensor([cosmo[l] for l in LABELS], dtype=torch.float) # graph labels, shape: num_nodes, num_features
    x = torch.tensor(peaks_vals.reshape(-1, 1), dtype=torch.float) # node features

    # making edges
    for key in range(len(peaks_hpinds)):
        neigh_hpinds = all_neigh_hpinds[peaks_hpinds[key]]
        apeaks_hpinds = np.intersect1d(peaks_hpinds, neigh_hpinds)
        apeaks_hpinds = apeaks_hpinds[apeaks_hpinds != peaks_hpinds[key]]
        apeaks_key = [peak_key(hpind) for hpind in apeaks_hpinds] # not using np.vectorize as apeaks is expected to be small
        
        edge_index = torch.cat((edge_index, torch.tensor(np.stack((np.repeat(key, len(apeaks_key)), apeaks_key)), dtype=torch.int)), dim=1)
        ra, dec = pix2ang[peaks_hpinds[key]]
        aradecs = pix2ang[apeaks_hpinds]
        ara, adec = aradecs[:, 0], aradecs[:, 1]
        seperations = np.sqrt((ara - ra)**2 + (adec - dec)**2) # flat sky approximation
        angles = np.arctan2(ara - ra, adec - dec)
        edge_attr = torch.cat((edge_attr, torch.tensor(np.stack((seperations, angles)).T, dtype=torch.float)))

    datapoint = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    gc.collect()
    # print(f"Processed {filename} {sc} {tb} in {time.time() - start} seconds", flush=True)
    return datapoint

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
        return os.path.join(GRAPHS_PATH(self.dataset_name), f"../scales.npy")
    
    @property
    def tomobin(self):
        return os.path.join(GRAPHS_PATH(self.dataset_name), f"../tomobins.npy")
    
    @property
    def healpy_preprocessed(self):
        return os.path.join(GRAPHS_PATH(self.dataset_name), f"../healpy_preprocessed.pkl")
    
    def write_scales_and_tomobins(self, scales, tomobins):
        np.save(self.scale, scales)
        np.save(self.tomobin, tomobins)

    def len(self):
        if SYSTEM_NAME == "perlmutter":
            return len(os.listdir(self.raw_dir)) - 1
        elif SYSTEM_NAME == "marmalade":
            with open(os.path.join(self.processed_dir, "done"), "r") as f:
                return int(f.read())  

    def __init__(self, dataset_name, scales, tomobins, nside=512, num_batches=NUM_BATCHES, field_name='k_sm_kE', radius=45, \
                overwrite=False, transform=None, pre_transform=None, pre_filter=None):
        # dataset name should have _tomobin***_scale*** in it
        self.dataset_name = dataset_name
        self.num_batches = num_batches
        self.field_name = field_name
        self.radius = radius / 60 / 180 * np.pi # input assumed in arcmins, converted to radians later
        self.nside = nside
        
        # we preprocess some healpy things to make stuff faster
        self.preprocess_healpy()
        
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
        self.write_scales_and_tomobins(scales, tomobins)
    
        # this super call calls process
        super().__init__(None, transform, pre_transform, pre_filter)

        self.loaded_batch_ind = None
        self.loaded_batch = None

    def preprocess_healpy(self):
        start = time.time()
        print("Preprocessing healpy", flush=True)
        nside = self.nside
        try:
            with open(self.healpy_preprocessed, 'rb') as fil:
                f = pickle.load(fil)
                self.all_hpinds = f['all_hpinds']
                self.all_hpvec = f['all_hpvec']
                self.all_neigh_hpinds = f['all_neigh_hpinds']
                self.pix2ang = f['pix2ang']
            print(f"Preprocessed healpy loaded in {time.time() - start} seconds", flush=True)
        except:
            print("Preprocessed healpy not found, generating", flush=True)
            self.all_hpinds = np.arange(hp.nside2npix(nside))
            self.all_hpvec = np.vstack(hp.pix2vec(nside, self.all_hpinds))
            self.all_neigh_hpinds = [hp.query_disc(nside, self.all_hpvec[:, i], self.radius, inclusive=True) for i in range(len(self.all_hpinds))]
            self.pix2ang = np.array([hp.pix2ang(nside, hpind, lonlat=True) for hpind in self.all_hpinds])
            with open(self.healpy_preprocessed, 'wb') as f:
                pickle.dump({'all_hpinds': self.all_hpinds, 'all_hpvec': self.all_hpvec, 'all_neigh_hpinds': self.all_neigh_hpinds, 'pix2ang': self.pix2ang}, f)
            print(f"Preprocessed healpy in {time.time() - start} seconds", flush=True)
        

    def process_one_file(self, filename):
        print(filename, flush=True)
        datapoints = []
        scales, tomobins = np.load(self.scale), np.load(self.tomobin)
        with open(os.path.join(self.raw_dir, filename), 'rb') as f:
            mmap, cosmo = pickle.load(f)
            all_peaks = mmap.peaks[self.field_name]
            del mmap
        
        for tb in tomobins:
            for sc in scales:
                start = time.time()
                # print(filename, sc, tb, flush=True)
                edge_index = torch.empty((2, 0), dtype=torch.int) # edges, in COO format, shape (2, num_edges)
                edge_attr = torch.empty((0, 2), dtype=torch.float) # edge features: separation, angle, shape (num_edges, num_features)
                
                peaks_hpinds = np.array(all_peaks[tb][f"{sc:.1f}"+"_loc"])
                peaks_vals = np.array(all_peaks[tb][f"{sc:.1f}"+"_val"])
                
                peak_key = lambda hpind: np.where(peaks_hpinds == hpind)[0][0]
                
                y = torch.tensor([cosmo[l] for l in LABELS], dtype=torch.float) # graph labels, shape: num_nodes, num_features
                x = torch.tensor(peaks_vals.reshape(-1, 1), dtype=torch.float) # node features
        
                # making edges
                for key in range(len(peaks_hpinds)):
                    neigh_hpinds = self.all_neigh_hpinds[peaks_hpinds[key]]
                    apeaks_hpinds = np.intersect1d(peaks_hpinds, neigh_hpinds)
                    apeaks_hpinds = apeaks_hpinds[apeaks_hpinds != peaks_hpinds[key]]
                    apeaks_key = [peak_key(hpind) for hpind in apeaks_hpinds] # not using np.vectorize as apeaks is expected to be small
                    
                    edge_index = torch.cat((edge_index, torch.tensor(np.stack((np.repeat(key, len(apeaks_key)), apeaks_key)), dtype=torch.int)), dim=1)
                    ra, dec = self.pix2ang[peaks_hpinds[key]]
                    aradecs = self.pix2ang[apeaks_hpinds]
                    ara, adec = aradecs[:, 0], aradecs[:, 1]
                    seperations = np.sqrt((ara - ra)**2 + (adec - dec)**2) # flat sky approximation
                    angles = np.arctan2(ara - ra, adec - dec)
                    edge_attr = torch.cat((edge_attr, torch.tensor(np.stack((seperations, angles)).T, dtype=torch.float)))
        
                datapoint = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
                datapoints.append(datapoint)
                gc.collect()
                # print(f"Processed {filename} {sc} {tb} in {time.time() - start} seconds", flush=True)
        print(f"Processed {filename} in {time.time() - start} seconds", flush=True)
        return datapoints
        
    
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
        
        gc.collect()
        
        pool = mp.Pool(nthreads)
        data_list = pool.map(self.process_one_file, filenames)
        pool.close()
        pool.join()
        
        gc.collect()
        
        # TODO pre filter or pre transform do not work with the datapoints setup
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]
        
        torch.save(data_list, self.processed_paths[batch_ind])
        print(f"Batch {batch_ind} processed in {time.time() - start} seconds", flush=True)
        return

    def process(self):
        if MPI_ON and False:
            run_count = 0
            comm = MPI.COMM_WORLD
            print(comm)
            while run_count < self.num_batches:
                if (run_count + comm.rank) < self.num_batches:
                    try:
                        self.process_one_batch(run_count + comm.rank)
                        # break
                    except:
                        pass
                run_count += comm.size
                gc.collect()
        else:
            for batch_ind in range(self.num_batches):
                self.process_one_batch(batch_ind)
                gc.collect()
                # break

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
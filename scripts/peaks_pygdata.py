import torch
from torch_geometric.data import Dataset, Data
import os, tqdm
import numpy as np

PEAKS_PATH = lambda dataset_name: f'/global/cfs/cdirs/des/shubh/graphs_weak_lensing/data/{dataset_name}/peaks/'
GRAPHS_PATH = lambda dataset_name: f'/global/cfs/cdirs/des/shubh/graphs_weak_lensing/data/{dataset_name}/graphs/'
NUM_BATCHES = 100

class Patches(Dataset):
    @property
    def raw_dir(self) -> str:
        return PEAKS_PATH(self.dataset_name)
    
    @property
    def processed_dir(self) -> str:
        return GRAPHS_PATH(self.dataset_name)

    @property
    def raw_file_names(self):
        return ["done"]

    @property
    def processed_file_names(self):
        return [f'data{ind}.pt' for ind in range(self.num_batches)]

    def len(self):
        return len(os.listdir(self.raw_dir)) - 1

    def __init__(self, dataset_name, num_batches=NUM_BATCHES, mp_on=True, overwrite=False, \
                    transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.num_batches = num_batches
        self.mp_on = mp_on
        
        if overwrite:
            for filename in os.listdir(self.processed_dir):
                os.remove(os.path.join(self.processed_dir, filename))

        filenames = os.listdir(self.raw_dir)
        filenames.remove("done")
        self.filenames_batched = np.array_split(filenames, self.num_batches)
        self.indices_batched = np.array_split(np.arange(len(filenames)), self.num_batches)
    
        super().__init__(None, transform, pre_transform, pre_filter)

        self.loaded_batch_ind = None
        self.loaded_batch = None

    def process_one_batch(self, batch_ind):
        if os.path.exists(self.processed_paths[batch_ind]):
            print(f"Batch {batch_ind} already processed")
            return
        print(f"Processing batch {batch_ind}")
        filenames = self.filenames_batched[batch_ind]
        data_list = []
        if self.mp_on:
            func = lambda x: x # no progress bar
        else:
            func = tqdm.tqdm
        for filename in func(filenames):
            data = np.load(os.path.join(self.raw_dir, filename))

            x = torch.tensor(data["peaks"]["val"].reshape(-1, 1), dtype=torch.float)
            y = torch.tensor(data["labels"], dtype=torch.float)
            edge_index = torch.tensor(data["edges"]["keys"], dtype=torch.int).T
            edge_attr = torch.tensor(np.stack((data["edges"]["sep"], \
                                                data["edges"]["ang"])), dtype=torch.float).T

            datapoint = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(datapoint)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        torch.save(data_list, self.processed_paths[batch_ind])
        return

    def process(self):
        if self.mp_on:
            import multiprocessing as mp
            pool = mp.Pool(mp.cpu_count())
            for _ in tqdm.tqdm(pool.imap_unordered(self.process_one_batch, \
                            range(self.num_batches)), total=self.num_batches):
                pass
            pool.close()
            pool.join()
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
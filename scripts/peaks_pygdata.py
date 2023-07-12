import torch
from torch_geometric.data import InMemoryDataset, Data
import os, tqdm
import numpy as np

PEAKS_PATH = lambda dataset_name: f'/global/cfs/cdirs/des/shubh/graphs_weak_lensing/data/{dataset_name}/peaks/'
GRAPHS_PATH = lambda dataset_name: f'/global/cfs/cdirs/des/shubh/graphs_weak_lensing/data/{dataset_name}/graphs/'

class Patches(InMemoryDataset):
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
        return ['data.pt']

    def __init__(self, dataset_name, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        super().__init__(None, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        for filename in tqdm.tqdm(os.listdir(self.raw_dir)):
            if filename == "done":
                continue
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

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
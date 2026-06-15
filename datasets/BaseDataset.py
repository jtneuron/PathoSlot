import copy
import os

import h5py
import numpy as np
import torch

from config.ConfigMixin import ConfigMixin


class BaseDataset(torch.utils.data.Dataset, ConfigMixin):
    def __init__(self, split):
        super().__init__()
        self.split = split
        self.data = {sample["id"]: sample for sample in self.split.data}
        self.ids = list(self.data.keys())
        self.num_folds = self.split.num_folds
        self.current_iter = None

    def get_subset(self, iteration, fold):
        subset = self.copy()
        subset.current_iter = iteration
        subset.data = {
            sample["id"]: sample
            for sample in self.split.data
            if sample["folds"][iteration] == fold
        }
        subset.ids = list(subset.data.keys())

        if hasattr(subset, "child_datasets"):
            for dataset_name, dataset in subset.child_datasets.items():
                if hasattr(dataset, "child_datasets"):
                    raise ValueError(f"Nested child datasets are not supported: {dataset_name}")
                subset.child_datasets[dataset_name] = dataset.get_subset(iteration, fold)

        return subset if subset.ids else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        raise NotImplementedError

    def collate_fn(self, batch):
        collated = {key: self.recursive_collate([d[key] for d in batch]) for key in batch[0].keys()}
        collated["current_iter"] = self.current_iter
        return collated

    def recursive_collate(self, items):
        if isinstance(items[0], torch.Tensor):
            return torch.stack(items, dim=0)
        if isinstance(items[0], np.ndarray):
            return np.stack(items, axis=0)
        if isinstance(items[0], dict):
            return {key: self.recursive_collate([d[key] for d in items]) for key in items[0].keys()}
        if isinstance(items[0], list):
            return [self.recursive_collate([d[i] for d in items]) for i in range(len(items[0]))]
        return items

    @staticmethod
    def load_h5(load_path, keys=None):
        if keys is not None and not isinstance(keys, list):
            raise TypeError("keys must be a list or None")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File {load_path} does not exist")

        try:
            with h5py.File(load_path, "r") as file:
                keys = list(file.keys()) if keys is None else keys
                assets = {key: file[key][:] for key in keys}
                attributes = {key: dict(file[key].attrs) for key in keys}
        except Exception as exc:
            raise RuntimeError(f"Error loading h5 file at {load_path}") from exc
        return assets, attributes

    def copy(self):
        return copy.deepcopy(self)

    def get_datasampler(self, sampler="random"):
        if sampler == "sequential":
            return torch.utils.data.SequentialSampler(self)
        if sampler == "random":
            return torch.utils.data.RandomSampler(self)
        raise NotImplementedError(f"Sampler type {sampler} not implemented")

    def get_dataloader(self, current_iter, fold, batch_size=None):
        subset_dataset = self.get_subset(current_iter, fold)
        if subset_dataset is None:
            return None

        return torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=len(subset_dataset) if batch_size is None else batch_size,
            sampler=subset_dataset.get_datasampler("random"),
            num_workers=0,
            collate_fn=subset_dataset.collate_fn,
        )

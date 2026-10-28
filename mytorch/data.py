from collections.abc import Callable
from typing import Optional, Protocol

import random

import numpy as np

import torch
from torch import Tensor


class Dataset(Protocol):
    """General dataset class"""
    samples: list[tuple]
    def __getitem__(self, i: int) -> tuple: ...
    def __len__(self) -> int: ...
    def split(
            self, i: int
            ) -> tuple['Dataset', 'Dataset']: ... 
    def subset(self, i: int) -> 'Dataset': ...


class BasicDataset(Dataset):
    """Basic dataset class without transforms"""
    def __init__(self, samples: list[tuple]) -> None:
        self.samples = samples


    def __getitem__(self, i: int) -> tuple:
        return self.samples[i]


    def __len__(self) -> int:
        return len(self.samples)


    def split(self, i: int) -> tuple[Dataset, Dataset]:
        return BasicDataset(self.samples[:i]), BasicDataset(self.samples[i:])


    def subset(self, i: int) -> Dataset:
        return BasicDataset(self.samples[:i])


class LoadDataset(Dataset):
    """Dataset class with data loading"""
    loader: Callable[[tuple], tuple]


    def __init__(
            self, samples: list[tuple], loader: Callable[[tuple], tuple]
            ) -> None:
        self.samples = samples
        self.loader = loader


    def __getitem__(self, i: int) -> tuple:
        return self.loader(self.samples[i])


    def __len__(self) -> int:
        return len(self.samples)


    def split(self, i: int) -> tuple[Dataset, Dataset]:
        return LoadDataset(self.samples[:i], self.loader),\
            LoadDataset(self.samples[i:], self.loader)


    def subset(self, i: int) -> Dataset:
        return LoadDataset(self.samples[:i], self.loader)


GPUTransform = Callable[[Tensor], Tensor]
Transform = Callable[[list[tuple]], tuple[Tensor]]


def seed_worker(worker_id: int) -> None:
    """Seed relevant libraries for the worker"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_worker_new(worker_id: int) -> None:
    """Seed relevant libraries for the worker"""
    worker_seed = torch.initial_seed() % (2**32-2)
    np.random.seed(worker_seed+1)
    random.seed(worker_seed+2)


def train_val_split(
        samples: list[tuple], shuffle_seed: Optional[int], val_frac: float
        ) -> tuple[list[tuple], list[tuple]]:
    """Split samples into two random parts"""
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(samples)
    train_size = int((1-val_frac) * len(samples))
    return samples[:train_size], samples[train_size:]

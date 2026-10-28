from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import cv2

import mytorch as my
from . import transforms


Loader = Callable[[tuple], tuple[np.ndarray, np.ndarray, int]]
Cropper = Union[transforms.TrainCropper, transforms.ValCropper]


class Noop:
    """Read skewed samples from memory"""
    def __call__(
            self, batch: tuple[np.ndarray, np.ndarray, int]
            ) -> tuple[np.ndarray, np.ndarray, int]:
        x, s_x, y, s = batch
        if not s:
            s_x = x.copy()
        return x, s_x, y


class FileLoad:
    """Read skewed samples from disk"""
    def __call__(
            self, batch: tuple[str, str, int]
            ) -> tuple[np.ndarray, np.ndarray, int]:
        """Load images"""
        x, s_x, y, s = batch
        x = cv2.imread(x)
        if not s:
            s_x = x.copy()
        else:
            s_x = cv2.imread(s_x)
        return x, s_x, y


class ImagesLoader:
    """
    Loader: load, crop
    """
    loader: Loader
    cropper: Cropper


    def __init__(self, cropper: Cropper, loader: Loader):
        self.cropper = cropper
        self.loader = loader


    def __call__(
            self, batch: tuple
            ) -> tuple[np.ndarray, np.ndarray, int]:
        x, s_x, y = self.loader(batch)
        x, s_x, y = self.cropper((x, s_x, y))
        x = np.flip(x, 2)
        s_x = np.flip(s_x, 2)
        x = x.transpose(2, 0, 1)
        s_x = s_x.transpose(2, 0, 1)
        return x, s_x, y


def init_imagenet_image_data(
        samples: list[tuple[str, str, int]],
        *, shuffle_seed: Optional[int], val_frac: float,
        size: int, up_size: int,
        bscale: float, tscale: float,
        bratio: float, tratio: float,
        y_dtype: Union[np.dtype, type],
        in_memory: bool,
        validation: bool = False
        ) -> tuple[
            my.data.Dataset, my.data.Dataset,
            my.data.Transform, my.data.Transform]:
    """Initialize ImageNet-like dataset with corresponding transforms"""
    # prepare datasets
    train_samples, val_samples =\
        my.data.train_val_split(samples, shuffle_seed, val_frac)
    val_cropper = transforms.ValCropper(size, size, up_size, up_size)

    # prepare transformations
    if validation:
        train_cropper = val_cropper
    else:
        train_cropper = transforms.TrainCropper(
            size, size, bscale, tscale, bratio, tratio)
    loader = Noop() if in_memory else FileLoad()
    train_dataset = my.data.LoadDataset(
        train_samples, ImagesLoader(train_cropper, loader))
    val_dataset = my.data.LoadDataset(
        val_samples, ImagesLoader(val_cropper, loader))

    # prepare collate
    val_transform = transforms.StdCollate(y_dtype)
    train_transform = val_transform
    return train_dataset, val_dataset, train_transform, val_transform

from typing import Union

import math
import random

import numpy as np
import cv2

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import mytorch as my


# transforms
def match_img(z: np.ndarray, vsize: int, hsize: int):
    """Resize and center crop image to match dimensions"""
    z_vsize, z_hsize, _ = z.shape
    if vsize * z_hsize > hsize * z_vsize:
        new_size = (int(vsize*z_hsize/z_vsize), vsize)
    else:
        new_size = (hsize, int(hsize*z_vsize/z_hsize))
    z = cv2.resize(z, new_size, interpolation=cv2.INTER_LINEAR)
    voff = (new_size[1] - vsize)//2
    hoff = (new_size[0] - hsize)//2
    z = z[voff:vsize+voff, hoff:hsize+hoff]
    return z


def gen_random_resized_crop(
        x_vsize: int, x_hsize: int,
        bscale: float, tscale: float,
        log_bratio: float, log_tratio: float
        ) -> tuple[int, int, int, int]:
    """Generate parameters for random resized crop"""
    area = x_vsize * x_hsize
    vsize, hsize = None, None
    for _ in range(10):
        new_area = area * random.uniform(bscale, tscale)
        ratio = 2**random.uniform(log_bratio, log_tratio)
        new_vsize = int(round(math.sqrt(new_area * ratio)))
        new_hsize = int(round(math.sqrt(new_area / ratio)))
        if 0 < new_vsize <= x_vsize and 0 < new_hsize <= x_hsize:
            vsize, hsize = new_vsize, new_hsize
            break
    if vsize is None:
        vsize, hsize = x_vsize, x_hsize
    vmax = x_vsize - vsize
    hmax = x_hsize - hsize
    vshift = random.randint(0, vmax)
    hshift = random.randint(0, hmax)
    return vsize, hsize, vshift, hshift


class TrainCropper:
    """
    Train cropper: random resized crop
    """
    vsize: int
    hsize: int
    bscale: float
    tscale: float
    bratio: float
    tratio: float


    def __init__(
            self,
            vsize: int, hsize: int,
            bscale: float, tscale: float,
            bratio: float, tratio: float):
        self.vsize = vsize
        self.hsize = hsize
        self.bscale = bscale
        self.tscale = tscale
        self.log_bratio = math.log2(bratio)
        self.log_tratio = math.log2(tratio)


    def __call__(
            self, sample: tuple[np.ndarray, np.ndarray, int]
            ) -> tuple[np.ndarray, np.ndarray, int]:
        x, s_x, y = sample
        x_vsize, x_hsize, _ = x.shape
        vsize, hsize, vshift, hshift = gen_random_resized_crop(
            x_vsize, x_hsize,
            self.bscale, self.tscale, self.log_bratio, self.log_tratio)
        flip = random.random() < 1/2
        x = self._x_transform(x, vsize, hsize, vshift, hshift, flip)
        s_x = self._x_transform(s_x, vsize, hsize, vshift, hshift, flip)
        return x, s_x, y


    def _x_transform(
            self, x: np.ndarray,
            vsize: int, hsize: int, vshift: int, hshift: int, flip: bool
            ) -> np.ndarray:
        x = x[vshift:vsize+vshift, hshift:hsize+hshift]
        # x = match_img(x, self.vsize, self.hsize)
        x = cv2.resize(
            x, (self.hsize, self.vsize), interpolation=cv2.INTER_LINEAR)
        if flip:
            x = np.flip(x, 1)
        return x


class ValCropper:
    """
    Val cropper: upscale, center crop
    """
    vsize: int
    hsize: int
    up_vsize: int
    up_hsize: int
    voff: int
    hoff: int


    def __init__(
            self, vsize: int, hsize: int,
            up_vsize: int, up_hsize: int):
        self.vsize = vsize
        self.hsize = hsize
        self.up_vsize = up_vsize
        self.up_hsize = up_hsize
        self.voff = (up_vsize - vsize)//2
        self.hoff = (up_hsize - hsize)//2


    def __call__(
            self, sample: tuple[np.ndarray, np.ndarray, int]
            ) -> tuple[np.ndarray, np.ndarray, int]:
        x, s_x, y = sample
        x = self._x_transform(x)
        s_x = self._x_transform(s_x)
        return x, s_x, y


    def _x_transform(self, x: np.ndarray) -> np.ndarray:
        x = match_img(x, self.up_vsize, self.up_hsize)
        x = x[self.voff:self.vsize+self.voff, self.hoff:self.hsize+self.hoff]
        return x


# collate
def xxy_batch_to_np(
        batch: list[tuple[np.ndarray, np.ndarray, int]], y_dtype: np.dtype
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """list of samples to tuple of numpy arrays (i.e., batching)"""
    x0, _, _ = batch[0]
    x = np.fromiter(
        (x for x, _, _ in batch),
        np.dtype((x0.dtype, x0.shape)), len(batch))
    s_x = np.fromiter(
        (x for _, x, _ in batch),
        np.dtype((x0.dtype, x0.shape)), len(batch))
    y = np.fromiter(
        (x for _, _, x in batch), y_dtype, len(batch))
    return x, s_x, y


class StdCollate:
    """Standard collate"""
    y_dtype: np.dtype


    def __init__(self, y_dtype: np.dtype):
        self.y_dtype = y_dtype


    def __call__(
            self, batch: list[tuple[np.ndarray, np.ndarray, int]]
            ) -> tuple[Tensor, Tensor, Tensor]:
        x, s_x, y = xxy_batch_to_np(batch, self.y_dtype)
        return tuple(map(torch.as_tensor, (x, s_x, y)))


# GPU transforms
class StdGPUXTransform:
    """Standard GPU transform of inputs: normalize and send to device"""
    rank: int
    means: Union[Tensor, float]
    stds: Union[Tensor, float]
    x_dtype: torch.dtype


    def __init__(
            self, rank: int,
            means: Union[Tensor, float], stds: Union[Tensor, float],
            x_dtype: torch.dtype = torch.float):
        self.rank = rank
        self.means = means
        self.stds = stds
        self.x_dtype = x_dtype


    def __call__(self, x: Tensor) -> Tensor:
        x = x.cuda(self.rank, non_blocking=True).to(self.x_dtype)
        return my.transforms.normalize(x, self.means, self.stds)


class StdGPUYTransform:
    """Standard GPU transform of labels: send to device"""
    rank: int
    y_dtype: torch.dtype


    def __init__(self, rank: int, y_dtype: torch.dtype = torch.long):
        self.rank = rank
        self.y_dtype = y_dtype


    def __call__(self, y: Tensor) -> Tensor:
        return y.cuda(self.rank, non_blocking=True).to(self.y_dtype)


def init_std_gpu_transforms(
        rank: int, init_dataloader: DataLoader, init_x_batch: Tensor,
        *, means: Union[Tensor, float], stds: Union[Tensor, float]
        ) -> tuple[my.data.GPUTransform, my.data.GPUTransform]:
    """Initialize standard GPU transforms"""
    if init_x_batch.ndim > 2:
        means = means.reshape((1, -1, 1, 1))
        stds = stds.reshape((1, -1, 1, 1))
    else:
        means = means.reshape((1, -1))
        stds = stds.reshape((1, -1))
    means = means.cuda(rank)
    stds = stds.cuda(rank)
    gpu_x_transform = StdGPUXTransform(rank, means, stds)
    gpu_y_transform = StdGPUYTransform(rank)
    return gpu_x_transform, gpu_y_transform

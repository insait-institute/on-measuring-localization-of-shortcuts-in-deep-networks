from typing import Any, Union

import numpy as np
import cv2

import torch
from torch import Tensor


# loaders
def copy_img(x: np.ndarray) -> np.ndarray:
    """Copies a tensor (to allow in-place transforms)"""
    return np.copy(x, order="C")


#transforms
def normalize(
        x: Tensor, means: Union[float, Tensor], stds: Union[float, Tensor]
        ) -> Tensor:
    """Normalize tensor"""
    x -= means
    x /= stds
    return x


# collate
def xy_batch_to_np(
        batch: list[tuple[np.ndarray, int]], y_dtype: np.dtype = np.uint8
        ) -> tuple[np.ndarray, np.ndarray]:
    """create two np.ndarray from tuple of lists"""
    x0, y0 = batch[0]
    x = np.fromiter(
        (x for x, _ in batch), np.dtype((x0.dtype, x0.shape)), len(batch))
    y = np.fromiter(
        (y for _, y in batch), y_dtype, len(batch))
    return x, y


# batch transforms
def cv_img_to_img(x: np.ndarray) -> np.ndarray:
    """HWC -> CHW"""
    return x.transpose(0, 3, 1, 2)


def gray_img_to_img(x: np.ndarray) -> np.ndarray:
    """HW -> CHW"""
    return np.expand_dims(x, 1)


def img_to_flat(x: np.ndarray) -> np.ndarray:
    """HWC -> array"""
    return np.reshape(x, (x.shape[0], -1))


# augmentations (could use in-place operations)

# skews (could use in-place operations)
def identity(x: Any):
    """Identity transform"""
    return x


def blend_int(x: np.ndarray, y: np.ndarray, inv_a: int) -> np.ndarray:
    """
    Blend x (in-place) and y (not modified).
    res = x + (y - x) / inv_a
    """
    z = np.maximum(x, y)
    w = z - y
    z -= x
    z //= inv_a
    w //= inv_a
    x -= w
    x += z
    return x


def blend(x: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    """
    Blend x (in-place) and y (not modified).
    res = (1-a) x + a y
    """
    x_double = x.astype(np.double)
    y_double = y.astype(np.double)
    np.copyto(x, ((1-a)*x_double + a*y_double).round().astype(np.uint8))
    return x

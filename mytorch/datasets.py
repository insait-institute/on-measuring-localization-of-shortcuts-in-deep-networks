import pickle

import numpy as np

from torchvision.datasets import MNIST, ImageFolder


def mnist(
        *, root: str, split: str = "train", download: bool = False
        ) -> list[tuple[np.ndarray, int]]:
    """MNIST dataset"""
    train = (split=="train")
    dataset = MNIST(root, train, download=download)
    data = np.ascontiguousarray(dataset.data.numpy())
    return list(zip(data, dataset.targets.tolist()))


def cifar10(
        *, root: str, split: str = "train"
        ) -> list[tuple[np.ndarray, int]]:
    """CIFAR-10 dataset (HWC, BGR format)"""
    if split=="train":
        batches = [None] * 5
        for i in range(1, 6):
            with open(f"{root}/cifar-10-batches-py/data_batch_{i}", "rb") as f:
                batches[i-1] = pickle.load(f, encoding="bytes")
        images = [
            image for i in range(5) for image in np.ascontiguousarray(
                np.flip(
                    batches[i][b"data"]\
                        .reshape((-1, 3, 32, 32))\
                        .transpose((0, 2, 3, 1)),
                    3))]
        labels = sum((batches[i][b"labels"] for i in range(5)), [])
    elif split=="test":
        with open(f"{root}/cifar-10-batches-py/test_batch", "rb") as f:
            test_batch = pickle.load(f, encoding="bytes")
        images = [
            image for image in np.ascontiguousarray(
                np.flip(
                    test_batch[b"data"]\
                        .reshape((-1, 3, 32, 32))\
                        .transpose((0, 2, 3, 1)),
                    3))]
        labels = test_batch[b"labels"]
    else:
        raise ValueError(f"Split {split} is not implemented for CIFAR-10")
    return list(zip(images, labels))


def cifar100(
        *, root: str, split: str = "train"
        ) -> list[tuple[np.ndarray, int]]:
    """CIFAR-100 dataset (HWC, BGR format)"""
    with open(f"{root}/cifar-100-python/{split}", "rb") as f:
        dataset = pickle.load(f, encoding="bytes")
    images = [
        image for image in np.ascontiguousarray(
            np.flip(
                dataset[b"data"]\
                    .reshape((-1, 3, 32, 32))\
                    .transpose((0, 2, 3, 1)),
                3))]
    labels = dataset[b"fine_labels"]
    return list(zip(images, labels))


def imagenet(
        *, root: str, split: str = "train") -> list[tuple[str, int]]:
    """ILSVRC dataset from Kaggle, validation folder fixed manually"""
    if split == "train":
        return ImageFolder(f"{root}/ILSVRC/Data/CLS-LOC/train").samples
    return ImageFolder(f"{root}/ILSVRC/Data/CLS-LOC/val-folder").samples

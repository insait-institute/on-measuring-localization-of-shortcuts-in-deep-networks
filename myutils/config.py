from typing import Any, Optional

import random
import copy

import numpy as np

import torch
import torchvision.models as tvm

import mytorch as my
from . import datasets, transforms, data, models, optim


# parameters
SAMPLES: dict[str, dict[str, Any]] = {
    "c10m-s": {
        "samples_fn": datasets.cifar_10_mnist,
        "samples_params": {
            "split": "train", "match_seed": 1, "s_size": 10, "s_a": 3/4}},
    "c10m-w": {
        "samples_fn": datasets.cifar_10_mnist,
        "samples_params": {
            "split": "train", "match_seed": 1, "s_size": 10, "s_a": 1/4}},
    "wb": {
        "samples_fn": datasets.waterbirds,
        "samples_params": {"split": "train", "match_seed": 2}},
    "ca": {
        "samples_fn": datasets.celeba_subsampled,
        "samples_params": {"split": "train+valid", "match_seed": 3}}}


SKEW: dict[str, dict[str, Any]] = {
    "c10m-c": {"skew_seed": 11, "s_freq": 127/128},
    "c10m-r": {"skew_seed": 11, "s_freq": 15/16},
    "wb-c": {"skew_seed": 13, "s_freq": 127/128},
    "wb-r": {"skew_seed": 13, "s_freq": 15/16},
    "ca-c": {"skew_seed": 13, "s_freq": 127/128},
    "ca-r": {"skew_seed": 13, "s_freq": 15/16}}


DATA: dict[str, dict[str, Any]] = {
    "c10m": {
        "data_fn": data.init_imagenet_image_data,
        "data_params": {
            "shuffle_seed": 21, "size": 224, "up_size": 224,
            "bscale": 0.8, "tscale": 1.0, "bratio": 3/4, "tratio": 4/3,
            "y_dtype": np.uint8, "in_memory": True}},
    "wb": {
        "data_fn": data.init_imagenet_image_data,
        "data_params": {
            "shuffle_seed": 22, "size": 224, "up_size": 256,
            "bscale": 0.7, "tscale": 1.0, "bratio": 3/4, "tratio": 4/3,
            "y_dtype": bool, "in_memory": False}},
    "ca": {
        "data_fn": data.init_imagenet_image_data,
        "data_params": {
            "shuffle_seed": 23, "size": 224, "up_size": 224,
            "bscale": 0.7, "tscale": 1.0, "bratio": 1, "tratio": 4/3,
            "y_dtype": bool, "in_memory": False}}}


DATALOADERS: dict[str, dict[str, Any]] = {
    "imagenet": {
        "train_dataloader_seed": None,
        "train_dataloader_params": {
            "batch_size": 2**6,
            "num_workers": 4, "prefetch_factor": 2,
            "pin_memory": True, "persistent_workers": True},
        "val_dataloader_params": {
            "batch_size": 2**7,
            "num_workers": 4, "prefetch_factor": 2,
            "pin_memory": True, "persistent_workers": True},
        "init_gpu_transforms": transforms.init_std_gpu_transforms,
        "gpu_transforms_params": {
            "means": torch.tensor([0.485, 0.456, 0.406]).mul_(255),
            "stds": torch.tensor([0.229, 0.224, 0.225]).mul_(255)}}}


MODELS: dict[str, dict[str, Any]] = {
    "dtt-s": {
        "init_model": models.init_deit,
        "model_params": {
            "model_seed": None, "num_classes": None,
            "model_type": 'deit_tiny_patch16_224',
            "patch_size": None,
            "model_load_params": {}}},
    "r18-s": {
        "init_model": models.init_resnet,
        "model_params": {
            "model_seed": None, "num_classes": None,
            "model_type": tvm.resnet18,
            "kernel_size": None, "stride": None,
            "model_load_params": {}}},
    "v11-s": {
        "init_model": models.init_vgg,
        "model_params": {
            "model_seed": None, "num_classes": None,
            "model_type": tvm.vgg11_bn, "fc_to_gap": True,
            "pool_size": (1, 1),
            "model_load_params": {}}},
    "dtt-f": {
        "init_model": models.init_deit,
        "model_params": {
            "model_seed": None, "num_classes": None,
            "model_type": 'deit_tiny_patch16_224',
            "patch_size": None,
            "model_load_params": {"pretrained": True}}},
    "dt8-f": {
        "init_model": models.init_deit8,
        "model_params": {
            "model_seed": None, "num_classes": None,
            "model_type": 'deit_tiny_patch16_224',
            "patch_size": None,
            "model_load_params": {"pretrained": True}}},
    "r18-f": {
        "init_model": models.init_resnet,
        "model_params": {
            "model_seed": None, "num_classes": None,
            "model_type": tvm.resnet18,
            "kernel_size": None, "stride": None,
            "model_load_params": {"weights": 'IMAGENET1K_V1'}}},
    "v11-f": {
        "init_model": models.init_vgg,
        "model_params": {
            "model_seed": None, "num_classes": None,
            "model_type": tvm.vgg11_bn, "fc_to_gap": True,
            "pool_size": (1, 1),
            "model_load_params": {"weights": 'IMAGENET1K_V1'}}},
    "cnt-f": {
        "init_model": models.init_convnext,
        "model_params": {
            "model_seed": None, "num_classes": None,
            "model_type": tvm.convnext_tiny,
            "model_load_params": {"weights": 'IMAGENET1K_V1'}}}
}


SPLITS: dict[str, dict[str, Any]] = {
    "test": {
        "split_fn": models.no_splits, "clean": True},
    "cl": {
        "split_fn": models.not_layer_splits, "clean": True},
    "ch": {
        "split_fn": models.head_splits, "clean": True},
    "sl": {
        "split_fn": models.not_layer_splits, "clean": False},
    "sh": {
        "split_fn": models.head_splits, "clean": False}}


OPTIMIZERS: dict[str, dict[str, Any]] = {
    "sgd-dtt-f": {
        "optimizer_fn": my.optim.init_sgd_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 2e-5, "weight_decay": 0.0001,
            "momentum": 0.9, "nesterov": True},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {
            "warmup_share": 0.02, "min_lr_factor": 5e-7,
            "scale_fn": my.optim.identity}},
    "flex-sgd-dtt-f": {
        "wd_factor": None, "lr_factor": None,
        "optimizer_fn": optim.init_flex_sgd_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 2e-5, "weight_decay": 0.0001,
            "momentum": 0.9, "nesterov": True},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {
            "warmup_share": 0.02, "min_lr_factor": 5e-7,
            "scale_fn": my.optim.identity}},
    "sgd-r18-f": {
        "optimizer_fn": my.optim.init_sgd_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 1e-4, "weight_decay": 0.0001,
            "momentum": 0.9, "nesterov": True},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {
            "warmup_share": 0.02, "min_lr_factor": 5e-7,
            "scale_fn": my.optim.identity}},
    "flex-sgd-r18-f": {
        "wd_factor": None, "lr_factor": None,
        "optimizer_fn": optim.init_flex_sgd_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 1e-4, "weight_decay": 0.0001,
            "momentum": 0.9, "nesterov": True},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {
            "warmup_share": 0.02, "min_lr_factor": 5e-7,
            "scale_fn": my.optim.identity}},
    "sgd-v11-f": {
        "optimizer_fn": my.optim.init_sgd_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 1e-4, "weight_decay": 0.0001,
            "momentum": 0.9, "nesterov": True},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {
            "warmup_share": 0.02, "min_lr_factor": 5e-7,
            "scale_fn": my.optim.identity}},
    "flex-sgd-v11-f": {
        "wd_factor": None, "lr_factor": None,
        "optimizer_fn": optim.init_flex_sgd_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 1e-4, "weight_decay": 0.0001,
            "momentum": 0.9, "nesterov": True},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {
            "warmup_share": 0.02, "min_lr_factor": 5e-7,
            "scale_fn": my.optim.identity}},
    "sgd-cnt-f": {
        "optimizer_fn": my.optim.init_sgd_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 1e-5, "weight_decay": 0.0001,
            "momentum": 0.9, "nesterov": True},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {
            "warmup_share": 0.02, "min_lr_factor": 5e-7,
            "scale_fn": my.optim.identity}},

    "adamw-dtt-f": {
        "optimizer_fn": my.optim.init_adamw_with_lr_scaling,
        "optimizer_params": {"lr_factor": 1e-5, "weight_decay": 0.01},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {"warmup_share": 0.02, "min_lr_factor": 1e-7}},
    "flex-adamw-dtt-f": {
        "wd_factor": None, "lr_factor": None,
        "optimizer_fn": optim.init_flex_adamw_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 1e-5, "weight_decay": 0.01,
            "lr_factors": None, "wd_factors": None},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {"warmup_share": 0.02, "min_lr_factor": 1e-7}},
    "adamw-r18-f": {
        "optimizer_fn": my.optim.init_adamw_with_lr_scaling,
        "optimizer_params": {"lr_factor": 1e-5, "weight_decay": 0.01},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {"warmup_share": 0.02, "min_lr_factor": 1e-7}},
    "flex-adamw-r18-f": {
        "wd_factor": None, "lr_factor": None,
        "optimizer_fn": optim.init_flex_adamw_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 1e-5, "weight_decay": 0.01,
            "lr_factors": None, "wd_factors": None},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {"warmup_share": 0.02, "min_lr_factor": 1e-7}},
    "adamw-v11-f": {
        "optimizer_fn": my.optim.init_adamw_with_lr_scaling,
        "optimizer_params": {"lr_factor": 1e-5, "weight_decay": 0.01},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {"warmup_share": 0.02, "min_lr_factor": 1e-7}},
    "flex-adamw-v11-f": {
        "wd_factor": None, "lr_factor": None,
        "optimizer_fn": optim.init_flex_adamw_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 1e-5, "weight_decay": 0.01,
            "lr_factors": None, "wd_factors": None},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {"warmup_share": 0.02, "min_lr_factor": 1e-7}},
    "adamw-cnt-f": {
        "optimizer_fn": my.optim.init_adamw_with_lr_scaling,
        "optimizer_params": {"lr_factor": 1e-5, "weight_decay": 0.01},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {"warmup_share": 0.02, "min_lr_factor": 1e-7}},

    "sgd-dtt-s": {
        "optimizer_fn": my.optim.init_sgd_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 2e-4, "weight_decay": 0.0001,
            "momentum": 0.9, "nesterov": True},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {
            "warmup_share": 0.05, "min_lr_factor": 5e-7,
            "scale_fn": my.optim.identity}},
    "sgd-r18-s": {
        "optimizer_fn": my.optim.init_sgd_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 5e-3, "weight_decay": 0.0001,
            "momentum": 0.9, "nesterov": True},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {
            "warmup_share": 0.05, "min_lr_factor": 2e-5,
            "scale_fn": my.optim.identity}},
    "sgd-v11-s": {
        "optimizer_fn": my.optim.init_sgd_with_lr_scaling,
        "optimizer_params": {
            "lr_factor": 5e-3, "weight_decay": 0.0001,
            "momentum": 0.9, "nesterov": True},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {
            "warmup_share": 0.05, "min_lr_factor": 2e-5,
            "scale_fn": my.optim.identity}},

    "adamw-dtt-s": {
        "optimizer_fn": my.optim.init_adamw_with_lr_scaling,
        "optimizer_params": {"lr_factor": 5e-5, "weight_decay": 0.01},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {"warmup_share": 0.05, "min_lr_factor": 1e-7}},
    "adamw-r18-s": {
        "optimizer_fn": my.optim.init_adamw_with_lr_scaling,
        "optimizer_params": {"lr_factor": 5e-3, "weight_decay": 0.01},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {"warmup_share": 0.05, "min_lr_factor": 5e-5}},
    "adamw-v11-s": {
        "optimizer_fn": my.optim.init_adamw_with_lr_scaling,
        "optimizer_params": {"lr_factor": 5e-3, "weight_decay": 0.01},
        "scheduler_fn": my.optim.init_cosine_scheduler,
        "scheduler_params": {"warmup_share": 0.05, "min_lr_factor": 5e-5}}}


NUM_CLASSES: dict[str, int] = {"c10m": 10, "wb": 2, "ca": 2}
EPOCHS: dict[str, int] = {
    "wb-s": 1000, "ca-s": 10, "c10m-s": 100,
    "wb-f": 20, "ca-f": 2, "c10m-f": 20}


# seeds before ConvNeXt was added
random.seed(42)
num_reps = 5
random_fn = lambda: random.randint(100, 9999)
OLD_LOADER_SEEDS: dict[str, tuple[int]] = {
    data_name: tuple(random_fn() for _ in range(num_reps))
    for data_name in DATA}
OLD_MODEL_SEEDS: dict[str, dict[str, tuple[int]]] = {
    data_name: {
        model_name: tuple(random_fn() for _ in range(num_reps))
        for model_name in list(MODELS)[:-1]}
    for data_name in DATA}
OLD_TRAIN_SEEDS: dict[str, dict[str, tuple[int]]] = {
    data_name: {
        model_name: tuple(random_fn() for _ in range(num_reps))
        for model_name in list(MODELS)[:-1]}
    for data_name in DATA}

# seeds after ConvNeXt was added
random.seed(42)
num_reps = 5
random_fn = lambda: random.randint(100, 9999)
LOADER_SEEDS: dict[str, tuple[int]] = {
    data_name: tuple(random_fn() for _ in range(num_reps))
    for data_name in DATA}
MODEL_SEEDS: dict[str, dict[str, tuple[int]]] = {
    data_name: {
        model_name: tuple(random_fn() for _ in range(num_reps))
        for model_name in MODELS}
    for data_name in DATA}
TRAIN_SEEDS: dict[str, dict[str, tuple[int]]] = {
    data_name: {
        model_name: tuple(random_fn() for _ in range(num_reps))
        for model_name in MODELS}
    for data_name in DATA}

# to use the old results
for key in OLD_LOADER_SEEDS:
    LOADER_SEEDS[key] = OLD_LOADER_SEEDS[key]
for key in OLD_MODEL_SEEDS:
    for key2 in OLD_MODEL_SEEDS[key]:
        MODEL_SEEDS[key][key2] = OLD_MODEL_SEEDS[key][key2]
for key in OLD_TRAIN_SEEDS:
    for key2 in OLD_TRAIN_SEEDS[key]:
        TRAIN_SEEDS[key][key2] = OLD_TRAIN_SEEDS[key][key2]


# parameter getters to avoid dicts modification
def get_samples(type: str, root: str) -> dict[str, Any]:
    """Get samples parameters"""
    samples = copy.deepcopy(SAMPLES[type])
    samples["samples_params"]["root"] = root
    return samples


def get_skew(type: str) -> dict[str, Any]:
    """Get skew parameters"""
    return copy.deepcopy(SKEW[type])


def get_data_obj(type: str, val_frac: float = 0.0) -> dict[str, Any]:
    """Get dataset and transforms parameters"""
    data = copy.deepcopy(DATA[type])
    data["data_params"]["val_frac"] = val_frac
    return data


def get_dataloader(
        type: str, seed: Optional[int], batch_size: Optional[int] = None
        ) -> dict[str, Any]:
    """Get DataLoader and GPU transforms parameters"""
    dataloader_params = copy.deepcopy(DATALOADERS[type])
    dataloader_params["train_dataloader_seed"] = seed
    if batch_size is not None:
        dataloader_params["train_dataloader_params"]["batch_size"] = batch_size
    return dataloader_params


def get_model(
        type: str, seed: Optional[int], num_classes: Optional[int]
        ) -> dict[str, Any]:
    """Get model parameters"""
    model_params = copy.deepcopy(MODELS[type])
    model_params["model_params"]["model_seed"] = seed
    model_params["model_params"]["num_classes"] = num_classes
    return model_params


def get_splits(type: str) -> dict[str, Any]:
    """Get split parameters"""
    return copy.deepcopy(SPLITS[type])


def get_optimizer(type: str) -> dict[str, Any]:
    """Get optimizer and scheduler parameters"""
    return copy.deepcopy(OPTIMIZERS[type])

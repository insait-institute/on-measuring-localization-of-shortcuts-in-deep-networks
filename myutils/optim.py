from collections.abc import Callable
from typing import Optional, Any

import math

from torch.utils.data import DataLoader
from torch import optim, nn

import mytorch as my


def init_flex_sgd_with_lr_scaling(
        dataloader: DataLoader, model: nn.Module, epochs: int,
        *, num_replicas: int = 1, lr_factor: float,
        scale_fn: Callable[[float], float] = lambda x: x,
        wd_factors: list[float], lr_factors: list[float],
        weight_decay: float,
        **optimizer_params
        ) -> optim.Optimizer:
    """Init SGD after (linear) scaling its lr by batch size"""
    lr = scale_fn(num_replicas * dataloader.batch_size) * lr_factor
    parts = sum(
        (nn.ModuleList(modality) for modality in model), nn.ModuleList([]))
    return optim.SGD(
        [
            {
                "params": parts[i].parameters(), "lr": lr * lr_factors[i],
                "weight_decay": weight_decay * wd_factors[i]}
            for i in range(len(parts))],
        **optimizer_params)


def init_flex_adamw_with_lr_scaling(
        dataloader: DataLoader, model: nn.Module, epochs: int,
        *, num_replicas: int = 1, lr_factor: float,
        scale_fn: Callable[[float], float] = math.sqrt,
        wd_factors: list[float], lr_factors: list[float],
        weight_decay: float,
        **optimizer_params
        ) -> optim.Optimizer:
    """Init AdamW after (square root) scaling its lr by batch size"""
    lr = scale_fn(num_replicas * dataloader.batch_size) * lr_factor
    parts = sum(
        (nn.ModuleList(modality) for modality in model), nn.ModuleList([]))
    return optim.AdamW(
        [
            {
                "params": parts[i].parameters(), "lr": lr * lr_factors[i],
                "weight_decay": weight_decay * wd_factors[i]}
            for i in range(len(parts))],
        **optimizer_params)


def init_optim(
        num_replicas: int, train_dataloader: DataLoader, models: nn.Module, epochs: int,
        *, optimizer_fn: Callable[..., optim.Optimizer],
        optimizer_params: dict[str, Any],
        scheduler_fn: Callable[..., my.optim.Scheduler],
        scheduler_params: dict[str, Any],
        max_epochs: Optional[int] = None
        ) -> tuple[list[optim.Optimizer], list[my.optim.Scheduler]]:
    """Initialize the same optimizer for all models"""
    if max_epochs is None:
        max_epochs = epochs
    optimizers = [
        optimizer_fn(
            train_dataloader, model, max_epochs,
            num_replicas=num_replicas, **optimizer_params)
        for model in models]
    schedulers = [
        scheduler_fn(
            train_dataloader, optimizer, max_epochs,
            num_replicas=num_replicas, **scheduler_params)
        for optimizer in optimizers]
    return optimizers, schedulers


def init_optim_exp(
        num_replicas: int, train_dataloader: DataLoader, models: nn.Module, epochs: int,
        *, optimizer_fn: Callable[..., optim.Optimizer],
        optimizer_params: dict[str, Any],
        scheduler_fn: Callable[..., my.optim.Scheduler],
        scheduler_params: dict[str, Any],
        max_epochs: Optional[int] = None,
        wd_factor: float, lr_factor: float
        ) -> tuple[list[optim.Optimizer], list[my.optim.Scheduler]]:
    """Intervene on optimizer parameters"""
    if max_epochs is None:
        max_epochs = epochs
    num_layers = sum((len(modality) for modality in models[0]))
    optimizers = []
    model_num = 0
    for i in range(num_layers):
        optimizer_params["wd_factors"] = [1.0] * num_layers
        optimizer_params["lr_factors"] = [1.0] * num_layers
        optimizer_params["wd_factors"][i] = wd_factor
        optimizer_params["lr_factors"][i] = lr_factor
        optimizers.append(
            optimizer_fn(
                train_dataloader, models[model_num], max_epochs,
                num_replicas=num_replicas, **optimizer_params))
        model_num += 1
    for i in range(2):
        optimizer_params["wd_factors"] = [1.0] * num_layers
        optimizer_params["lr_factors"] = [1.0] * num_layers
        optimizers.append(
            optimizer_fn(
                train_dataloader, models[model_num], max_epochs,
                num_replicas=num_replicas, **optimizer_params))
        model_num += 1
    schedulers = [
        scheduler_fn(
            train_dataloader, optimizer, max_epochs,
            num_replicas=num_replicas, **scheduler_params)
        for optimizer in optimizers]
    return optimizers, schedulers

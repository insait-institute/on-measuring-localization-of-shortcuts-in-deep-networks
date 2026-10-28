from collections.abc import Callable, Iterable
from typing import Any, Optional

import os
import copy
import pickle

import numpy as np
from tqdm.auto import trange

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

import mytorch as my
from . import config, datasets, models, optim


def get_val_losses(
        dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        models: list[nn.Module],
        loss_fn: Callable[[Tensor, Tensor], Tensor] = my.optim.error
        ) -> tuple[list[float], list[float]]:
    """Get validation loss of models"""
    for model in models:
        model.eval()
    with torch.no_grad():
        val_losses, s_val_losses = [0] * len(models), [0] * len(models)
        for x, s_x, y in dataloader:
            x = gpu_x_transform(x)
            s_x = gpu_x_transform(s_x)
            y = gpu_y_transform(y)
            for i, model in enumerate(models):
                val_losses[i] += loss_fn(model(x), y)
                s_val_losses[i] += loss_fn(model(s_x), y)
        val_losses = torch.stack(val_losses) / len(dataloader.dataset)
        s_val_losses = torch.stack(s_val_losses) / len(dataloader.dataset)
        torch.distributed.all_reduce(val_losses, torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(s_val_losses, torch.distributed.ReduceOp.SUM)
        val_losses = val_losses.tolist()
        s_val_losses = s_val_losses.tolist()
    for model in models:
        model.train()
    return val_losses, s_val_losses


def print_val_err(
        rank: int, dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        models: list[nn.Module],
        model_names: list[str], epoch: int,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = my.optim.error
        ) -> None:
    """Print validation error of models"""
    val_losses, s_val_losses = get_val_losses(
        dataloader, gpu_x_transform, gpu_y_transform,
        models, loss_fn)
    if rank == 0:
        print(f"Epoch: {epoch:>3}")
        for name, val_loss, s_val_loss\
            in zip(model_names, val_losses, s_val_losses):
            print(
                f"    {name:<12}",
                f"Err: {val_loss:>5.1%}  S Err: {s_val_loss:>5.1%}")


def noop(*x) -> None:
    """Noop validation epoch/checkpoint action"""
    return None


def compute_grad(
        x: Tensor, y: Tensor, model: nn.Module, grad_layers: nn.Module,
        const_layers: nn.Module, fixed_layers: nn.Module
        ) -> None:
    """Compute gradient for sub-part of model"""
    grad_layers.requires_grad_(True)
    const_layers.requires_grad_(False)
    fixed_layers.requires_grad_(False)
    c_means, c_vars = my.utils.save_bn_stats(const_layers)
    f_means, f_vars = my.utils.save_bn_stats(fixed_layers)
    loss = my.optim.cross_entropy(model(x), y)
    loss.backward()
    my.utils.load_bn_stats(const_layers, c_means, c_vars)
    my.utils.load_bn_stats(fixed_layers, f_means, f_vars)


def train(
        rank: int, num_replicas: int,
        train_dataloader: DataLoader, val_dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        c_models: list[nn.Module], s_models: list[nn.Module],
        models: list[nn.Module],
        model_names: list[str],
        c_layers: list[nn.Module], s_layers: list[nn.Module],
        f_layers: list[nn.Module],
        optimizers: list[torch.optim.Optimizer],
        schedulers: list[my.optim.Scheduler],
        epochs: int,
        checkpoint_path: str, checkpoint_interval: int,
        checkpoint_fn: Callable[[dict, str], None],
        val_interval: int, val_fn: Callable,
        deterministic: bool, benchmark: bool, train_seed: Optional[int],
        zero_val: bool, range_fn: Callable[[int], Iterable]
        ) -> None:
    """Train and save seperate sub-models and anchor model"""
    # my.utils.set_deterministic(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    my.utils.set_all_device_seed(train_seed)
    val_resid = epochs % val_interval
    checkpoint_resid = epochs % checkpoint_interval
    seed_shift = 0
    if rank == 0 and not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
    for model in models:
        model.train()
    for optimizer in optimizers:
        optimizer.zero_grad()

    if zero_val:
        val_fn(
            rank, val_dataloader, gpu_x_transform, gpu_y_transform,
            models, model_names, 0)
        if rank == 0:
            for name, model in zip(model_names, models):
                checkpoint_fn(
                    model.state_dict(), f"{checkpoint_path}/{name}-e{0}.pt")
    if rank == 0:
        epoch_iter = range_fn(epochs)
    else:
        epoch_iter = range(epochs)
    for epoch in epoch_iter:
        if epoch > 0 and epoch % val_interval == val_resid:
            val_fn(
                rank, val_dataloader, gpu_x_transform, gpu_y_transform,
                models, model_names, epoch)
        if rank == 0 and epoch > 0\
            and epoch % checkpoint_interval == checkpoint_resid:
            for name, model in zip(model_names, models):
                checkpoint_fn(
                    model.state_dict(), f"{checkpoint_path}/{name}-e{epoch}.pt")
        for i, (x, s_x, y) in enumerate(train_dataloader):
            x = gpu_x_transform(x)
            s_x = gpu_x_transform(s_x)
            y = gpu_y_transform(y)
            for c_model, s_model, c_layer, s_layer, f_layer, optimizer, scheduler in\
                zip(c_models, s_models, c_layers, s_layers, f_layers, optimizers, schedulers):
                if c_layer:
                    torch.cuda.manual_seed(train_seed+seed_shift+i)
                    compute_grad(x, y, c_model, c_layer, s_layer, f_layer)
                if s_layer:
                    torch.cuda.manual_seed(train_seed+seed_shift+i)
                    compute_grad(s_x, y, s_model, s_layer, c_layer, f_layer)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        seed_shift += len(train_dataloader)
    val_fn(
        rank, val_dataloader, gpu_x_transform, gpu_y_transform,
        models, model_names, epochs)
    if rank == 0:
        for name, model in zip(model_names, models):
            checkpoint_fn(
                model.state_dict(), f"{checkpoint_path}/{name}-e{epochs}.pt")
    for scheduler in schedulers:
        scheduler.reset()


def init_train(
        rank: int,
        num_replicas: int,
        train_dataset: my.data.Dataset, val_dataset: my.data.Dataset,
        train_transform: my.data.Transform, val_transform: my.data.Transform,
        *, train_dataloader_seed: int,
        train_dataloader_params: dict[str, Any],
        val_dataloader_params: dict[str, Any],
        init_gpu_transforms: Callable[
            ..., tuple[my.data.GPUTransform, my.data.GPUTransform]],
        gpu_transforms_params: dict[str, Any],
        init_model: Callable[..., nn.Module],
        model_params: dict[str, Any],
        split_fn: Callable[
            [nn.Module, bool],
            tuple[
                list[nn.Module], list[str],
                list[nn.Module], list[nn.Module], list[nn.Module]]],
        clean: bool,
        optim_fn: Callable[..., tuple[list[torch.optim.Optimizer], list[my.optim.Scheduler]]],
        optim_params: dict[str, Any],
        epochs: int,
        checkpoint_path: str,
        checkpoint_interval: Optional[int] = None,
        checkpoint_fn: Callable[[dict, str], None] = torch.save,
        val_interval: Optional[int] = None, 
        val_fn: Callable = print_val_err,
        deterministic: bool = False, benchmark: bool = True,
        train_seed: Optional[int] = None,
        zero_val: bool = False, range_fn: Callable[[int], Iterable] = trange,
        ) -> tuple:
    """Initialize distributed training objects"""
    # prepare dataloaders
    generator = my.utils.init_generator(train_dataloader_seed+rank, None)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_transform,
        sampler=DistributedSampler(
            train_dataset, num_replicas, rank, seed=train_dataloader_seed),
        worker_init_fn=my.data.seed_worker_new,
        generator=generator,
        **train_dataloader_params)
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=val_transform,
        sampler=DistributedSampler(val_dataset, num_replicas, rank),
        **val_dataloader_params)
    init_dataloader = DataLoader(
        train_dataset,
        collate_fn=val_transform)
    init_x_batch = next(iter(init_dataloader))[0]

    # prepare gpu transforms
    gpu_x_transform, gpu_y_transform = init_gpu_transforms(
        rank, init_dataloader, init_x_batch, **gpu_transforms_params)
    init_x_batch = gpu_x_transform(init_x_batch)

    # prepare model
    model = init_model(
        rank, init_dataloader, gpu_x_transform, gpu_y_transform, init_x_batch,
        **model_params)
    model.cuda(rank)
    models, model_names, c_layers, s_layers, f_layers =\
        split_fn(model, clean=clean)

    # prepare optimizers
    optimizers, schedulers = optim_fn(
        num_replicas, train_dataloader, models, epochs, **optim_params)

    # prepare ddp
    c_models = []
    for model, c_layer, s_layer, f_layer in\
        zip(models, c_layers, s_layers, f_layers):
        if not c_layer:
            c_models.append(None)
            continue
        c_layer.requires_grad_(True)
        s_layer.requires_grad_(False)
        f_layer.requires_grad_(False)
        c_models.append(DistributedDataParallel(model, device_ids=[rank]))
    s_models = []
    for model, c_layer, s_layer, f_layer in\
        zip(models, c_layers, s_layers, f_layers):
        if not s_layer:
            s_models.append(None)
            continue
        s_layer.requires_grad_(True)
        c_layer.requires_grad_(False)
        f_layer.requires_grad_(False)
        s_models.append(DistributedDataParallel(model, device_ids=[rank]))

    # prepare training params
    if val_interval is None:
        val_interval = (epochs-1) // 4 + 1
    if checkpoint_interval is None:
        checkpoint_interval = (epochs-1) // 8 + 1

    return (
        rank, num_replicas,
        train_dataloader, val_dataloader,
        gpu_x_transform, gpu_y_transform,
        c_models, s_models, models,
        model_names, c_layers, s_layers, f_layers,
        optimizers, schedulers, epochs,
        checkpoint_path, checkpoint_interval, checkpoint_fn,
        val_interval, val_fn,
        deterministic, benchmark, train_seed, zero_val, range_fn)


def train_process(
        rank: int, num_replicas: int,
        port: int,
        working_dir: str,
        samples_name: str, model_name: str, optimizer_name: str,
        epochs: int, samples: list[tuple]
        ) -> None:
    """Distributed training process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=num_replicas)

    sub_pos = samples_name.find("-")
    if sub_pos != -1:
        data_name = samples_name[:samples_name.find("-")]
    else:
        data_name = samples_name
    full_optimizer_name = f"{optimizer_name}-{model_name}"
    train_name = model_name[model_name.find("-")+1:]
    if epochs <= 0:
        epochs = config.EPOCHS[f"{data_name}-{train_name}"]
    data_params = config.get_data_obj(data_name, 0.0)
    for freq_name in ["c", "r"]:
        full_samples_name = f"{samples_name}-{freq_name}"
        skew_params = config.get_skew(f"{data_name}-{freq_name}")
        skewed_samples = datasets.add_skew_flag(samples, **skew_params)
        data_obj = data_params["data_fn"](skewed_samples, **data_params["data_params"])
        for split_name in ["c", "s"]:
            if rank == 0:
                print(
                    f"Samples: {full_samples_name}, Model: {model_name},",
                    f"Split: {split_name}")
            for loader_seed, model_seed, train_seed in zip(
                config.LOADER_SEEDS[data_name],
                config.MODEL_SEEDS[data_name][model_name],
                config.TRAIN_SEEDS[data_name][model_name]):
                if rank == 0:
                    print(f"Model seed: {model_seed}")
                train_params = {
                    **config.get_dataloader(
                        "imagenet", loader_seed, 2**8//num_replicas),
                    **config.get_model(
                        model_name, model_seed, config.NUM_CLASSES[data_name]),
                    "optim_fn": optim.init_optim,
                    "optim_params": config.get_optimizer(full_optimizer_name),
                    "split_fn": models.all_splits, "clean": split_name=="c",
                    "epochs": epochs,
                    "checkpoint_path":
                        f"{working_dir}/checkpoints/{full_samples_name}-{model_name}-{optimizer_name}/s{model_seed}",
                    "checkpoint_interval": epochs,
                    "val_interval": epochs, "val_fn": noop,
                    "deterministic": True, "benchmark": False,
                    "train_seed": train_seed,
                    "zero_val": False, "range_fn": trange}
                train_obj = init_train(
                    rank, num_replicas, *data_obj, **train_params)
                train(*train_obj)

    destroy_process_group()


def test_process(
        rank: int, num_replicas: int, port: int, repeats: int,
        working_dir: str,
        samples_name: str, model_name: str, optim_params: dict[str, Any],
        epochs: int, samples: list[tuple]
        ) -> None:
    """Distributed process for testing hyperparameters"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=num_replicas)

    sub_pos = samples_name.find("-")
    if sub_pos != -1:
        data_name = samples_name[:samples_name.find("-")]
    else:
        data_name = samples_name
    train_name = model_name[model_name.find("-")+1:]
    if epochs <= 0:
        epochs = config.EPOCHS[f"{data_name}-{train_name}"]
    data_params = config.get_data_obj(data_name, 0.1)
    full_samples_name = f"{samples_name}-c"
    skew_params = config.get_skew(f"{data_name}-c")
    skewed_samples = datasets.add_skew_flag(samples, **skew_params)
    data_obj = data_params["data_fn"](skewed_samples, **data_params["data_params"])
    if rank == 0:
        print(
            f"Samples: {full_samples_name}, Model: {model_name}")
    for loader_seed, model_seed, train_seed in zip(
        config.LOADER_SEEDS[data_name][:repeats],
        config.MODEL_SEEDS[data_name][model_name],
        config.TRAIN_SEEDS[data_name][model_name]):
        if rank == 0:
            print(f"Model seed: {model_seed}")
        train_params = {
            **config.get_dataloader(
                "imagenet", loader_seed, 2**8//num_replicas),
            **config.get_model(
                model_name, model_seed, config.NUM_CLASSES[data_name]),
            "optim_fn": optim.init_optim,
            "optim_params": optim_params,
            "split_fn": models.no_splits, "clean": True,
            "epochs": epochs,
            "checkpoint_path": f"{working_dir}/checkpoints/tmp",
            "checkpoint_interval": epochs, "checkpoint_fn": noop,
            "val_interval": epochs,
            "deterministic": True, "benchmark": False,
            "train_seed": train_seed,
            "zero_val": False, "range_fn": trange}
        train_obj = init_train(rank, num_replicas, *data_obj, **train_params)
        train(*train_obj)

    destroy_process_group()


def flex_train_process(
        rank: int, num_replicas: int,
        port: int,
        working_dir: str,
        samples_name: str, model_name: str,
        optimizer_name: str, freq_name: str,
        epochs: int, samples: list[tuple]
        ) -> None:
    """Distributed training process for interventions"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=num_replicas)

    sub_pos = samples_name.find("-")
    if sub_pos != -1:
        data_name = samples_name[:samples_name.find("-")]
    else:
        data_name = samples_name
    full_optimizer_name = f"{optimizer_name}-{model_name}"
    train_name = model_name[model_name.find("-")+1:]
    if epochs <= 0:
        epochs = config.EPOCHS[f"{data_name}-{train_name}"]
    data_params = config.get_data_obj(data_name, 0.0)
    full_samples_name = f"{samples_name}-{freq_name}"
    skew_params = config.get_skew(f"{data_name}-{freq_name}")
    skewed_samples = datasets.add_skew_flag(samples, **skew_params)
    data_obj = data_params["data_fn"](skewed_samples, **data_params["data_params"])
    for exp_name, lr_fact, wd_fact in (
        ("lr-up", 3, 1), ("lr-down", 1/3, 1),
        ("wd-up", 1, 10), ("wd-down", 1, 1/10)):
        if rank == 0:
            print(
                f"Samples: {full_samples_name}, Model: {model_name},",
                f"LR Factor: {lr_fact:.1f}, WD Factor: {wd_fact:.1f}")
        for loader_seed, model_seed, train_seed in zip(
            config.LOADER_SEEDS[data_name],
            config.MODEL_SEEDS[data_name][model_name],
            config.TRAIN_SEEDS[data_name][model_name]):
            if rank == 0:
                print(f"Model seed: {model_seed}")
            train_params = {
                **config.get_dataloader(
                    "imagenet", loader_seed, 2**8//num_replicas),
                **config.get_model(
                    model_name, model_seed, config.NUM_CLASSES[data_name]),
                "optim_fn": optim.init_optim_exp,
                "optim_params": config.get_optimizer(full_optimizer_name),
                "split_fn": models.exp_splits, "clean": False,
                "epochs": epochs,
                "checkpoint_path":
                    f"{working_dir}/checkpoints/"\
                    f"{full_samples_name}-{model_name}-{optimizer_name}-{exp_name}/"\
                    f"s{model_seed}",
                "checkpoint_interval": epochs,
                "val_interval": epochs, "val_fn": noop,
                "deterministic": True, "benchmark": False,
                "train_seed": train_seed,
                "zero_val": False, "range_fn": trange}
            train_params["optim_params"]["wd_factor"] = wd_fact
            train_params["optim_params"]["lr_factor"] = lr_fact
            train_obj = init_train(
                rank, num_replicas, *data_obj, **train_params)
            train(*train_obj)

    destroy_process_group()


def freeze_train_process(
        rank: int, num_replicas: int,
        port: int,
        working_dir: str,
        samples_name: str, model_name: str,
        optimizer_name: str, freq_name: str,
        epochs: int, samples: list[tuple]
        ) -> None:
    """Distributed training process for the freezing intervention"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=num_replicas)

    sub_pos = samples_name.find("-")
    if sub_pos != -1:
        data_name = samples_name[:samples_name.find("-")]
    else:
        data_name = samples_name
    full_optimizer_name = f"{optimizer_name}-{model_name}"
    train_name = model_name[model_name.find("-")+1:]
    if epochs <= 0:
        epochs = config.EPOCHS[f"{data_name}-{train_name}"]
    data_params = config.get_data_obj(data_name, 0.0)
    # data_params = config.get_data_obj(data_name, 0.1)
    full_samples_name = f"{samples_name}-{freq_name}"
    skew_params = config.get_skew(f"{data_name}-{freq_name}")
    skewed_samples = datasets.add_skew_flag(samples, **skew_params)
    data_obj = data_params["data_fn"](skewed_samples, **data_params["data_params"])
    if rank == 0:
        print(f"Samples: {full_samples_name}, Model: {model_name}")
    for loader_seed, model_seed, train_seed in zip(
        config.LOADER_SEEDS[data_name],
        config.MODEL_SEEDS[data_name][model_name],
        config.TRAIN_SEEDS[data_name][model_name]):
        if rank == 0:
            print(f"Model seed: {model_seed}")
        train_params = {
            **config.get_dataloader(
                "imagenet", loader_seed, 2**8//num_replicas),
            **config.get_model(
                model_name, model_seed, config.NUM_CLASSES[data_name]),
            "optim_fn": optim.init_optim,
            "optim_params": config.get_optimizer(full_optimizer_name),
            "split_fn": models.last_layer, "clean": False,
            "epochs": 1,
            "checkpoint_path": f"{working_dir}/checkpoints/tmp",
            "checkpoint_interval": 1, "checkpoint_fn": noop,
            "val_interval": 1,
            "val_fn": noop,
            "deterministic": True, "benchmark": False,
            "train_seed": train_seed,
            "zero_val": False, "range_fn": trange}
        train_params["optim_params"]["optimizer_params"]["lr_factor"]\
            *= (epochs / 20 * config.NUM_CLASSES[data_name])
        train_params["optim_params"]["scheduler_params"]["min_lr_factor"]\
            = train_params["optim_params"]["optimizer_params"]["lr_factor"]
        train_params["optim_params"]["scheduler_params"]["warmup_share"]\
            = 0
        train_obj = init_train(
            rank, num_replicas, *data_obj, **train_params)
        train(*train_obj)

        train_params = {
            **config.get_dataloader(
                "imagenet", loader_seed, 2**8//num_replicas),
            "init_model": models.load_model,
            "model_params": {"model": train_obj[8][0]},
            "optim_fn": optim.init_optim,
            "optim_params": config.get_optimizer(full_optimizer_name),
            "split_fn": models.no_splits, "clean": False,
            "epochs": 1,
            "checkpoint_path": f"{working_dir}/checkpoints/tmp",
            "checkpoint_interval": 1, "checkpoint_fn": noop,
            "val_interval": 1,
            "val_fn": noop,
            "deterministic": True, "benchmark": False,
            "train_seed": train_seed,
            "zero_val": False, "range_fn": trange}
        train_params["optim_params"]["optimizer_params"]["lr_factor"]\
            *= epochs/5000
        train_params["optim_params"]["scheduler_params"]["min_lr_factor"] = min(
            train_params["optim_params"]["scheduler_params"]["min_lr_factor"],
            train_params["optim_params"]["optimizer_params"]["lr_factor"])
        train_obj = init_train(
            rank, num_replicas, *data_obj, **train_params)
        train(*train_obj)

        train_params = {
            **config.get_dataloader(
                "imagenet", loader_seed, 2**8//num_replicas),
            "init_model": models.load_model,
            "model_params": {"model": train_obj[8][0]},
            "optim_fn": optim.init_optim,
            "optim_params": config.get_optimizer(full_optimizer_name),
            "split_fn": models.freeze_splits, "clean": False,
            "epochs": epochs,
            "checkpoint_path":
                f"{working_dir}/checkpoints/"\
                f"{full_samples_name}-{model_name}-{optimizer_name}-freeze/"\
                f"s{model_seed}",
            "checkpoint_interval": epochs,
            "val_interval": epochs,
            "val_fn": noop,
            "deterministic": True, "benchmark":False,
            "train_seed": train_seed,
            "zero_val": False, "range_fn": trange}
        train_obj = init_train(
            rank, num_replicas, *data_obj, **train_params)
        train(*train_obj)

    destroy_process_group()


def get_sub_names(split_name: str, model: nn.Module) -> list[str]:
    """Generate sub-model names from split name"""
    num_layers = sum(len(block) for block in model)
    a0, a1 = "clean", "skewed"
    if split_name[0] == "c":
        a0, a1 = a1, a0
    if split_name[1] == "h":
        sub_names = [a0]\
            + [f"{split_name[0]}_{i}:{num_layers}" for i in range(1, num_layers)]\
            + [a1]
    else:
        sub_names = [a0]\
            + [f"{split_name[0]}_-{i}" for i in range(num_layers)]\
            + [a1]
    return sub_names


def init_val_dataloader(
        rank: int, num_replicas: int,
        dataset: my.data.Dataset, transform: my.data.Transform,
        *, train_dataloader_seed: int,
        train_dataloader_params: dict[str, Any],
        val_dataloader_params: dict[str, Any],
        init_gpu_transforms: Callable[
            ..., tuple[my.data.GPUTransform, my.data.GPUTransform]],
        gpu_transforms_params: dict[str, Any]
        ) -> tuple[DataLoader, my.data.GPUTransform, my.data.GPUTransform, torch.Tensor]:
    """Initialize validation DataLoader and GPU transforms"""
    dataloader = DataLoader(
        dataset,
        collate_fn=transform,
        sampler=DistributedSampler(dataset, num_replicas, rank),
        **val_dataloader_params)
    init_x_batch = next(iter(dataloader))[0]

    # prepare gpu transforms
    gpu_x_transform, gpu_y_transform = init_gpu_transforms(
        rank, dataloader, init_x_batch, **gpu_transforms_params)
    init_x_batch = gpu_x_transform(init_x_batch)

    return dataloader, gpu_x_transform, gpu_y_transform, init_x_batch


def validate(
        dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        models: list[nn.Module],
        num_seeds: int,
        loss_fn: Callable[[Tensor, Tensor], Tensor]
        ) -> tuple[np.ndarray, np.ndarray]:
    """Get validation losses and reshape them into (model, seed)-format"""
    val_losses, s_val_losses = get_val_losses(
        dataloader, gpu_x_transform, gpu_y_transform, models, loss_fn)
    return (
        np.array(val_losses).reshape(-1, num_seeds),
        np.array(s_val_losses).reshape(-1, num_seeds))


def init_validate(
        rank: int, num_replicas: int,
        dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        model: nn.Module,
        checkpoint_path: str,
        samples_name: str, model_name: str, optimizer_name: str,
        split_name: str,
        *, loss_fn: Callable[[Tensor, Tensor], Tensor] = my.optim.error
        ) -> tuple:
    """Initialize distributed validation objects"""
    sub_pos = samples_name.find("-")
    data_name = samples_name[:samples_name.find("-")]
    train_name = model_name[model_name.find("-")+1:]

    epochs = config.EPOCHS[f"{data_name}-{train_name}"]
    sub_names = get_sub_names(split_name, model)
    model_seeds = config.MODEL_SEEDS[data_name][model_name]

    models = [copy.deepcopy(model) for _ in model_seeds for _ in sub_names]
    for i, sub_name in enumerate(sub_names):
        for j, model_seed in enumerate(model_seeds):
            models[i*len(model_seeds) + j].load_state_dict(torch.load(
                f"{checkpoint_path}/{samples_name}-{model_name}-{optimizer_name}/s{model_seed}/{sub_name}-e{epochs}.pt",
                map_location="cpu", weights_only=True))
            models[i*len(model_seeds) + j].cuda(rank)

    return (
        dataloader, gpu_x_transform, gpu_y_transform,
        models, len(model_seeds), loss_fn)


def val_process(
        rank: int, num_replicas: int, port: int,
        working_dir: str, split_name: str
        ) -> None:
    """Distributed validation process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=num_replicas)

    checkpoint_path = f"{working_dir}/checkpoints"
    val_losses = {}
    s_val_losses = {}
    
    for i, samples_name in enumerate(["c10m-s", "c10m-w", "ca", "wb"]):
        sub_pos = samples_name.find("-")
        if sub_pos != -1:
            data_name = samples_name[:samples_name.find("-")]
        else:
            data_name = samples_name
        samples_params\
            = config.get_samples(samples_name, f"{working_dir}/datasets")
        samples_params["samples_params"]["split"] = "test"
        samples\
            = samples_params["samples_fn"](**samples_params["samples_params"])
        data_params = config.get_data_obj(data_name, 1.0)
        skewed_samples = datasets.add_skew_flag(samples, i+30, 1.0)
        _, dataset, _, transform = data_params["data_fn"](
            skewed_samples, **data_params["data_params"])
        dataloader_params = config.get_dataloader(
            "imagenet", 0, 2**8//num_replicas)
        dataloader, gpu_x_transform, gpu_y_transform, init_x_batch\
            = init_val_dataloader(
                rank, num_replicas, dataset, transform,
                **dataloader_params)


        for model_name in ["dtt-s", "dtt-f", "r18-s", "r18-f", "v11-s", "v11-f", "cnt-f"]:
            if model_name == "dtt-s" and samples_name == "wb":
                continue
            model_params = config.get_model(
                model_name, 0, config.NUM_CLASSES[data_name])
            model = model_params["init_model"](
                rank, dataloader, gpu_x_transform, gpu_y_transform, init_x_batch,
                **model_params["model_params"])

            for freq_name in ["c", "r"]:
                full_samples_name = f"{samples_name}-{freq_name}"

                for optimizer_name in ["adamw", "sgd"]:
                    if rank == 0:
                        print(
                            f"Split: {split_name},",
                            f"Samples: {full_samples_name},",
                            f"Model: {model_name},",
                            f"Optimzier: {optimizer_name}")

                    val_obj = init_validate(
                        rank, num_replicas,
                        dataloader, gpu_x_transform, gpu_y_transform,
                        model, checkpoint_path,
                        full_samples_name, model_name, optimizer_name,
                        split_name)
                    val_loss, s_val_loss = validate(*val_obj)
                    val_losses[f"{full_samples_name}-{model_name}-{optimizer_name}"]\
                        = val_loss
                    s_val_losses[f"{full_samples_name}-{model_name}-{optimizer_name}"]\
                        = s_val_loss

    if rank == 0:
        with open(f'{checkpoint_path}/{split_name}-err.pickle', 'wb') as f:
            pickle.dump(val_losses, f, pickle.HIGHEST_PROTOCOL)
        with open(f'{checkpoint_path}/{split_name}-err-s.pickle', 'wb') as f:
            pickle.dump(s_val_losses, f, pickle.HIGHEST_PROTOCOL)

    destroy_process_group()


def init_flex_validate(
        rank: int, num_replicas: int,
        dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        model: nn.Module,
        checkpoint_path: str,
        samples_name: str, model_name: str, optimizer_name: str, exp_name: str,
        clean: bool = False,
        *, loss_fn: Callable[[Tensor, Tensor], Tensor] = my.optim.error
        ) -> tuple:
    """Initialize distributed validation objects for internventions"""
    sub_pos = samples_name.find("-")
    data_name = samples_name[:samples_name.find("-")]
    train_name = model_name[model_name.find("-")+1:]

    epochs = config.EPOCHS[f"{data_name}-{train_name}"]

    num_layers = sum(len(modality) for modality in model)
    a_name = "clean" if clean else "skewed"
    b_name = "clean" if not clean else "skewed"
    sub_names\
        = [f"{a_name[0]}_{i}" for i in range(num_layers)]\
            + [a_name]
            # + [f"{a_name[0]}_{i}:{i+2}" for i in range(num_layers-1)]\
            # + [a_name, b_name]
    model_seeds = config.MODEL_SEEDS[data_name][model_name]

    models = [copy.deepcopy(model) for _ in model_seeds for _ in sub_names]
    for i, sub_name in enumerate(sub_names):
        for j, model_seed in enumerate(model_seeds):
            models[i*len(model_seeds) + j].load_state_dict(torch.load(
                f"{checkpoint_path}/{samples_name}-{model_name}-{optimizer_name}-{exp_name}/s{model_seed}/{sub_name}-e{epochs}.pt",
                map_location="cpu", weights_only=True))
            models[i*len(model_seeds) + j].cuda(rank)

    return (
        dataloader, gpu_x_transform, gpu_y_transform,
        models, len(model_seeds), loss_fn)


def flex_val_process(
        rank: int, num_replicas: int, port: int,
        working_dir: str,
        optimizer_name: str, freq_name: str
        ) -> None:
    """Distributed validation process for interventions"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=num_replicas)

    checkpoint_path = f"{working_dir}/checkpoints"
    val_losses = {}
    s_val_losses = {}
    
    for i, samples_name in enumerate(["c10m-s", "ca", "wb"]):
        sub_pos = samples_name.find("-")
        if sub_pos != -1:
            data_name = samples_name[:samples_name.find("-")]
        else:
            data_name = samples_name
        samples_params\
            = config.get_samples(samples_name, f"{working_dir}/datasets")
        samples_params["samples_params"]["split"] = "test"
        samples\
            = samples_params["samples_fn"](**samples_params["samples_params"])
        data_params = config.get_data_obj(data_name, 1.0)
        skewed_samples = datasets.add_skew_flag(samples, i+30, 1.0)
        _, dataset, _, transform = data_params["data_fn"](
            skewed_samples, **data_params["data_params"])
        dataloader_params = config.get_dataloader(
            "imagenet", 0, 2**8//num_replicas)
        dataloader, gpu_x_transform, gpu_y_transform, init_x_batch\
            = init_val_dataloader(
                rank, num_replicas, dataset, transform,
                **dataloader_params)


        for model_name in ["dtt-f", "r18-f", "v11-f"]:
            model_params = config.get_model(
                model_name, 0, config.NUM_CLASSES[data_name])
            model = model_params["init_model"](
                rank, dataloader, gpu_x_transform, gpu_y_transform, init_x_batch,
                **model_params["model_params"])

            full_samples_name = f"{samples_name}-{freq_name}"

            for exp_name in ("lr-up", "lr-down", "wd-up", "wd-down"):
                if rank == 0:
                    print(
                        f"Exp: {exp_name},",
                        f"Samples: {full_samples_name},",
                        f"Model: {model_name},",
                        f"Optimzier: {optimizer_name}")
                val_obj = init_flex_validate(
                    rank, num_replicas,
                    dataloader, gpu_x_transform, gpu_y_transform,
                    model, checkpoint_path,
                    full_samples_name, model_name, optimizer_name,
                    exp_name)
                val_loss, s_val_loss = validate(*val_obj)
                val_losses[f"{full_samples_name}-{model_name}-{optimizer_name}-{exp_name}"]\
                    = val_loss
                s_val_losses[f"{full_samples_name}-{model_name}-{optimizer_name}-{exp_name}"]\
                    = s_val_loss

    if rank == 0:
        with open(f'{checkpoint_path}/exp-err.pickle', 'wb') as f:
            pickle.dump(val_losses, f, pickle.HIGHEST_PROTOCOL)
        with open(f'{checkpoint_path}/exp-err-s.pickle', 'wb') as f:
            pickle.dump(s_val_losses, f, pickle.HIGHEST_PROTOCOL)

    destroy_process_group()


def init_freeze_validate(
        rank: int, num_replicas: int,
        dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        model: nn.Module,
        checkpoint_path: str,
        samples_name: str, model_name: str, optimizer_name: str,
        clean: bool = False,
        *, loss_fn: Callable[[Tensor, Tensor], Tensor] = my.optim.error
        ) -> tuple:
    """Initialize distributed validation objects for the freezing intervention"""
    sub_pos = samples_name.find("-")
    data_name = samples_name[:samples_name.find("-")]
    train_name = model_name[model_name.find("-")+1:]

    epochs = config.EPOCHS[f"{data_name}-{train_name}"]

    num_layers = sum(len(modality) for modality in model)
    a_name = "clean" if clean else "skewed"
    sub_names\
        = [f"{a_name[0]}_{i}" for i in range(num_layers)] + [a_name]
    model_seeds = config.MODEL_SEEDS[data_name][model_name]

    models = [copy.deepcopy(model) for _ in model_seeds for _ in sub_names]
    for i, sub_name in enumerate(sub_names):
        for j, model_seed in enumerate(model_seeds):
            models[i*len(model_seeds) + j].load_state_dict(torch.load(
                f"{checkpoint_path}/{samples_name}-{model_name}-{optimizer_name}-freeze/s{model_seed}/{sub_name}-e{epochs}.pt",
                map_location="cpu", weights_only=True))
            models[i*len(model_seeds) + j].cuda(rank)

    return (
        dataloader, gpu_x_transform, gpu_y_transform,
        models, len(model_seeds), loss_fn)


def freeze_val_process(
        rank: int, num_replicas: int, port: int,
        working_dir: str,
        optimizer_name: str, freq_name: str
        ) -> None:
    """Distributed validation process for the freezing intervention"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=num_replicas)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint_path = f"{working_dir}/checkpoints"
    val_losses = {}
    s_val_losses = {}
    
    for i, samples_name in enumerate(["c10m-s", "ca", "wb"]):
        sub_pos = samples_name.find("-")
        if sub_pos != -1:
            data_name = samples_name[:samples_name.find("-")]
        else:
            data_name = samples_name
        samples_params\
            = config.get_samples(samples_name, f"{working_dir}/datasets")
        samples_params["samples_params"]["split"] = "test"
        samples\
            = samples_params["samples_fn"](**samples_params["samples_params"])
        data_params = config.get_data_obj(data_name, 1.0)
        skewed_samples = datasets.add_skew_flag(samples, i+30, 1.0)
        _, dataset, _, transform = data_params["data_fn"](
            skewed_samples, **data_params["data_params"])
        dataloader_params = config.get_dataloader(
            "imagenet", 0, 2**8//num_replicas)
        dataloader, gpu_x_transform, gpu_y_transform, init_x_batch\
            = init_val_dataloader(
                rank, num_replicas, dataset, transform,
                **dataloader_params)


        for model_name in ["dtt-f", "r18-f", "v11-f"]:
            model_params = config.get_model(
                model_name, 0, config.NUM_CLASSES[data_name])
            model = model_params["init_model"](
                rank, dataloader, gpu_x_transform, gpu_y_transform, init_x_batch,
                **model_params["model_params"])

            full_samples_name = f"{samples_name}-{freq_name}"

            if rank == 0:
                print(
                    f"Samples: {full_samples_name},",
                    f"Model: {model_name},",
                    f"Optimzier: {optimizer_name}")
            val_obj = init_freeze_validate(
                rank, num_replicas,
                dataloader, gpu_x_transform, gpu_y_transform,
                model, checkpoint_path,
                full_samples_name, model_name, optimizer_name)
            val_loss, s_val_loss = validate(*val_obj)
            val_losses[f"{full_samples_name}-{model_name}-{optimizer_name}-freeze"]\
                = val_loss
            s_val_losses[f"{full_samples_name}-{model_name}-{optimizer_name}-freeze"]\
                = s_val_loss

    if rank == 0:
        with open(f'{checkpoint_path}/fr-err.pickle', 'wb') as f:
            pickle.dump(val_losses, f, pickle.HIGHEST_PROTOCOL)
        with open(f'{checkpoint_path}/fr-err-s.pickle', 'wb') as f:
            pickle.dump(s_val_losses, f, pickle.HIGHEST_PROTOCOL)

    destroy_process_group()

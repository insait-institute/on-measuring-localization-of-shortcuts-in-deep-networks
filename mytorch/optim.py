from typing import Any, Protocol
from collections.abc import Callable

import math
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch import optim, nn


# loss
def cross_entropy(x: Tensor, y: Tensor, reduce: int = 1) -> Tensor:
    """
    Cross entropy loss
    0 -> no reduce, 1 -> mean, 2 -> sum
    """
    return torch._C._nn.cross_entropy_loss(x, y, None, reduce, -100, 0.0)


def bin_cross_entropy(x: Tensor, y: Tensor, reduce: int = 1) -> Tensor:
    """
    Binary cross entropy loss
    0 -> no reduce, 1 -> mean, 2 -> sum
    """
    return torch.binary_cross_entropy_with_logits(x, y, None, None, reduce)


def mean_squared_error(x: Tensor, y: Tensor, reduce: int = 1) -> Tensor:
    """
    Mean squared error
    0 -> no reduce, 1 -> mean, 2 -> sum
    """
    return torch._C._nn.mse_loss(x, y, reduce)


def error(x: Tensor, y: Tensor) -> Tensor:
    """Error function"""
    return (x.argmax(1) != y).sum()


def bin_error(x: Tensor, y: Tensor) -> Tensor:
    """Binary error function"""
    return ((x > 0) != y).sum()


def topk_error(x: Tensor, y: Tensor, k: int) -> Tensor:
    """Top-k error function"""
    pred_y = x.topk(k)[1]
    return len(y) - (y.unsqueeze(1) == pred_y).sum()


# optimizer
def identity(x: float) -> float:
    """Identity function"""
    return x


def init_sgd_with_lr_scaling(
        dataloader: DataLoader, model: nn.Module, epochs: int,
        *, num_replicas: int = 1, lr_factor: float,
        scale_fn: Callable[[float], float] = identity,
        **optimizer_params
        ) -> optim.Optimizer:
    """Init SGD after (linear) scaling its lr by batch size"""
    lr = scale_fn(num_replicas * dataloader.batch_size) * lr_factor
    return optim.SGD(model.parameters(), lr=lr, **optimizer_params)


def init_adamw_with_lr_scaling(
        dataloader: DataLoader, model: nn.Module, epochs: int,
        *, num_replicas: int = 1, lr_factor: float,
        scale_fn: Callable[[float], float] = math.sqrt,
        **optimizer_params
        ) -> optim.Optimizer:
    """Init AdamW after (square root) scaling its lr by batch size"""
    lr = scale_fn(num_replicas * dataloader.batch_size) * lr_factor
    return optim.AdamW(model.parameters(), lr=lr, **optimizer_params)


#schedulers
class Scheduler(Protocol):
    """Basic scheduler class"""
    param_groups: list[dict[str, Any]]
    def step(self) -> None: ...


class LinearScheduler(Scheduler):
    """Linear scheduler with warmup"""
    param_groups: list[dict[str, Any]]
    factor: float
    shift: float
    inc: float
    dec: float


    def __init__(
            self, param_groups: list[dict[str, Any]], factor: float,
            shift: float, inc: float, dec: float) -> None:
        self.param_groups = param_groups
        self.factor = factor
        self.shift = shift
        self.inc = inc
        self.dec = dec


    def step(self) -> None:
        for group in self.param_groups:
            group["lr"] = self.factor * group["initial_lr"]
        if self.factor + self.inc / 2 > 1: #check for achieving max_lr
            self.shift = self.dec
        self.factor += self.shift


    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.param_groups]


    def reset(self) -> None:
        self.factor = self.inc
        self.shift = self.inc
        self.step()


    def state_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor, "shift": self.shift,
            "inc": self.inc, "dec": self.dec}


    @classmethod
    def from_params(
        cls, optimizer: optim.Optimizer, num_warmup_steps: int,
        num_training_steps: int) -> 'LinearScheduler':
        param_groups = optimizer.param_groups
        factor = 1 / (num_warmup_steps + 1)
        shift = factor
        inc = factor
        dec = -1 / (num_training_steps - num_warmup_steps)
        self = cls(param_groups, factor, shift, inc, dec)
        for group in self.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.step()
        return self


    @classmethod
    def from_state_dict(
        cls, optimizer: optim.Optimizer, state: dict[str, Any]
        ) -> 'LinearScheduler':
        cls(optimizer.param_groups, **state)


    def load_state_dict(self, state: dict[str, Any]) -> 'LinearScheduler':
        self.factor = state["factor"]
        self.shift = state["shift"]
        self.inc = state["inc"]
        self.dec = state["dec"]


class CosineScheduler(Scheduler):
    """Linear scheduler with warmup"""
    param_groups: list[dict[str, Any]]
    factor: float
    shift: float
    inc: float
    dec: float
    min_lr: float
    trans: Callable[[float], float]


    def __init__(
            self, param_groups: list[dict[str, Any]],
            factor: float, shift: float,
            inc: float, dec: float, min_lr: float,
            trans: str
            ) -> None:
        self.param_groups = param_groups
        self.factor = factor
        self.shift = shift
        self.inc = inc
        self.dec = dec
        self.min_lr = min_lr
        if trans == "linear":
            self.trans = self._linear_factor
        else:
            self.trans = self._cosine_factor


    def step(self) -> None:
        for group in self.param_groups:
            group["lr"] = self.trans(group["initial_lr"])
        if 2 * self.factor + self.inc > math.pi:
            self.shift = self.dec
            self.trans = self._cosine_factor
        self.factor += self.shift


    def _linear_factor(self, initial_lr: float) -> float:
        factor = self.factor * 2 / math.pi
        lr = factor * (initial_lr + self.min_lr)
        return lr


    def _cosine_factor(self, initial_lr: float) -> float:
        factor = (math.sin(self.factor) + 1) / 2
        lr = factor * initial_lr + self.min_lr
        return lr


    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.param_groups]


    def reset(self) -> None:
        self.factor = self.inc
        self.shift = self.inc
        self.trans = self._linear_factor
        self.step()


    def state_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor, "shift": self.shift,
            "inc": self.inc, "dec": self.dec,
            "min_lr": self.min_lr,
            "trans": "linear" if self.trans == self._linear_factor else "cosine"}


    @classmethod
    def from_params(
        cls, optimizer: optim.Optimizer, num_warmup_steps: int,
        num_training_steps: int, min_lr: float) -> 'CosineScheduler':
        param_groups = optimizer.param_groups
        factor = math.pi / (2 * (num_warmup_steps + 1))
        shift = factor
        inc = factor
        dec = -math.pi / (num_training_steps - num_warmup_steps)
        self = cls(param_groups, factor, shift, inc, dec, min_lr, "linear")
        for group in self.param_groups:
            group.setdefault("initial_lr", max(group["lr"]-min_lr, 0))
        self.step()
        return self


    @classmethod
    def from_state_dict(
        cls, optimizer: optim.Optimizer, state: dict[str, Any]
        ) -> 'CosineScheduler':
        cls(optimizer.param_groups, **state)


    def load_state_dict(self, state: dict[str, Any]) -> 'CosineScheduler':
        self.factor = state["factor"]
        self.shift = state["shift"]
        self.inc = state["inc"]
        self.dec = state["dec"]
        self.trans = state["trans"]
        self.min_lr = state["min_lr"]


def init_linear_scheduler(
        dataloader: DataLoader, optimizer: optim.Optimizer, epochs: int,
        *, warmup_share: float) -> LinearScheduler:
    """Simple initializer for LinearScheduler"""
    num_trainig_steps = epochs * len(dataloader)
    num_warmup_steps = int(warmup_share * num_trainig_steps)
    return LinearScheduler.from_params(
        optimizer, num_warmup_steps, num_trainig_steps)


def init_cosine_scheduler(
        dataloader: DataLoader, optimizer: optim.Optimizer, epochs: int,
        *, num_replicas: int = 1,
        warmup_share: float, min_lr_factor: float = 0.0,
        scale_fn: Callable[[float], float] = math.sqrt
        ) -> CosineScheduler:
    """Simple initializer for CosineScheduler"""
    num_trainig_steps = epochs * len(dataloader)
    num_warmup_steps = int(warmup_share * num_trainig_steps)
    min_lr = scale_fn(num_replicas * dataloader.batch_size) * min_lr_factor
    return CosineScheduler.from_params(
        optimizer, num_warmup_steps, num_trainig_steps, min_lr)

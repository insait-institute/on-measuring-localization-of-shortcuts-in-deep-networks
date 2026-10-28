from typing import Optional, Any
from collections.abc import Callable

from collections import deque
import random

import numpy as np

import torch
from torch import Tensor
from torch import nn


def timeit(
        fn: Callable[[], Any],
        repeat: int = 2**8, groups: int = 2**2) -> float:
    """
    timeit for torch cuda functions.
    Aim for more than 10ms, when choosing repeat
    """
    def timed_fn(fn: Callable[[], Any]):
        """Capture for timing"""
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = tuple(timed_fn(fn) for _ in range(repeat))
    groups = min(groups, repeat)
    min_times = tuple(
        min(times[repeat*i//groups:repeat*(i+1)//groups])\
            for i in range(groups))
    t0, t1 = min(min_times), max(min_times) # shows variance in estimates
    return t0, t1, (t1 - t0) / t0


def set_all_seed(seed: Optional[int]) -> None:
    """Sets seed for all pytorch related random generators"""
    if seed is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_all_device_seed(seed: Optional[int]) -> None:
    """Sets seed for all pytorch related random generators"""
    if seed is not None:
        torch.cuda.manual_seed(seed)
    np.random.seed((2*seed+1)%(2**32))
    random.seed((4*seed+3)%(2**32))


def set_deterministic(deterministic: bool) -> None:
    """Force determenistic pytorch behavior"""
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = (not deterministic)


def set_torch_seed(seed: Optional[int]) -> None:
    """Sets seed for torch generators"""
    if seed is not None:
        torch.manual_seed(seed)


def init_generator(
        seed: Optional[int], device: Optional[torch.device]
        ) -> torch.Generator:
    """Init torch generator with seed"""
    g = torch.Generator(device)
    if seed is not None:
        g.manual_seed(seed)
    return g


def get_std_device() -> torch.device:
    """Alias for the one-liner below"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_bn_stats(model: nn.Module) -> tuple[deque, deque]:
    """Save BatchNorm2D statistics"""
    r_means = deque()
    r_vars = deque()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            r_means.append(module.running_mean)
            module.running_mean = None
            r_vars.append(module.running_var)
            module.running_var = None
    return r_means, r_vars


def load_bn_stats(model: nn.Module, r_means: deque, r_vars: deque) -> None:
    """Restore BatchNorm2D statistics"""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean = r_means.popleft()
            module.running_var = r_vars.popleft()


batched_dot = torch.vmap(torch.dot)


def sq_scale(x: Tensor) -> Tensor:
    """Sum of squares"""
    x = x.flatten()
    return torch.dot(x, x)

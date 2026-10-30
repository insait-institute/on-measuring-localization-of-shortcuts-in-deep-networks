from collections.abc import Callable
from typing import Any, Optional

import copy

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.models import VGG, ResNet, ConvNeXt
import timm
from timm.models import VisionTransformer

import mytorch as my


def vgg_to_sequential(model: VGG, fc_to_gap: bool) -> nn.Module:
    """
    Transform torchvision VGG to a layer-wise format
    """
    features = []
    last_conv_ind = 0
    for i in range(1, len(model.features)):
        if isinstance(model.features[i], nn.Conv2d):
            features.append(nn.Sequential(*model.features[last_conv_ind:i]))
            last_conv_ind = i
    features.append(
        nn.Sequential(*model.features[last_conv_ind:], model.avgpool))

    layers = []
    last_pool_ind = 0
    for i, feature in enumerate(features):
        if isinstance(feature[-1], nn.MaxPool2d):
            layers.append(nn.Sequential(*features[last_pool_ind:i+1]))
            last_pool_ind = i+1
    layers.append(nn.Sequential(*features[last_pool_ind:]))
    # layers.append(nn.Sequential(features[-1]))

    if fc_to_gap:
        return nn.Sequential(
            nn.Sequential(*layers),
            nn.Sequential(
                nn.Sequential(nn.Flatten(), model.classifier[6])))
    return nn.Sequential(
        nn.Sequential(*layers),
        nn.Sequential(
            nn.Sequential(
                nn.Sequential(nn.Flatten(), *model.classifier[:3]),
                nn.Sequential(*model.classifier[3:6])),
            nn.Sequential(model.classifier[6])))


def init_vgg(
        rank: int,
        init_dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        init_x_batch: Tensor,
        *, pool_size: Optional[tuple[int, int]],
        num_classes: Optional[int],
        model_seed: Optional[int],
        model_type: Callable[..., VGG], fc_to_gap: bool = False,
        model_load_params: dict[str, Any]
        ) -> nn.Module:
    """Initialize VGG"""
    my.utils.set_torch_seed(model_seed)
    model = model_type(**model_load_params)
    if fc_to_gap:
        if pool_size is not None:
            model.avgpool = nn.AdaptiveAvgPool2d(pool_size)
        size = model.avgpool.output_size[0] * model.avgpool.output_size[1]
        if num_classes is not None:
            model.classifier[-1] = nn.Linear(512 * size, num_classes)
    else:
        if pool_size is not None:
            model.avgpool = nn.AdaptiveAvgPool2d(pool_size)
            size = pool_size[0] * pool_size[1]
            model.classifier[0] = nn.Linear(512 * size, 4096)
        if num_classes is not None:
            model.classifier[-1] = nn.Linear(4096, num_classes)
    model = vgg_to_sequential(model, fc_to_gap)
    return model


def resnet_to_sequential(model: ResNet) -> nn.Module:
    """
    Transform torchvision ResNet to a layer-wise format
    """
    return nn.Sequential(
        nn.Sequential(
            nn.Sequential(
                nn.Sequential(
                    model.conv1, model.bn1, model.relu, model.maxpool)),
            model.layer1,
            model.layer2,
            model.layer3,
            nn.Sequential(
                *model.layer4[:-1],
                nn.Sequential(model.layer4[-1], model.avgpool))),
        nn.Sequential(
            nn.Sequential(nn.Flatten(), model.fc)))


def init_resnet(
        rank: int,
        init_dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        init_x_batch: Tensor,
        *, kernel_size: Optional[tuple[int, int]],
        stride: Optional[tuple[int, int]],
        num_classes: Optional[int],
        model_seed: Optional[int],
        model_type: Callable[..., ResNet],
        model_load_params: dict[str, Any]
        ) -> nn.Module:
    """Initialize ResNet"""
    my.utils.set_torch_seed(model_seed)
    model = model_type(**model_load_params)
    if kernel_size is not None:
        model.conv1 = nn.Conv2d(
            init_x_batch.shape[1], 64, kernel_size,
            stride, (kernel_size[0]//2, kernel_size[1]//2), bias=False)
    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = resnet_to_sequential(model)
    return model


class PosEmbed(nn.Module):
    """Positional embedding module copy-pasted from timm implementation"""
    cls_token: Optional[nn.Parameter]
    reg_token: Optional[nn.Parameter]
    pos_embed: Optional[nn.Parameter]
    pos_drop: nn.Dropout
    no_embed_class: bool


    def __init__(
            self,
            cls_token: Optional[nn.Parameter],
            reg_token: Optional[nn.Parameter],
            pos_embed: Optional[nn.Parameter],
            pos_drop: nn.Dropout,
            no_embed_class: bool):
        super().__init__()
        self.cls_token = cls_token
        self.reg_token = reg_token
        self.pos_embed = pos_embed
        self.pos_drop = pos_drop
        self.no_embed_class = no_embed_class


    def forward(self, x: Tensor) -> Tensor:
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        pos_embed = self.pos_embed
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed
        return self.pos_drop(x)


class PoolToken(nn.Module):
    """Pool token module (default version)"""
    def __init__(self):
        super().__init__()


    def forward(self, x: Tensor) -> Tensor:
        return x[:, 0]


def deit_to_sequential(model: VisionTransformer) -> nn.Module:
    """
    Transform timm DeiT to a layer-wise format
    """
    num_blocks = len(model.blocks)
    return nn.Sequential(
        nn.Sequential(
            nn.Sequential(
                model.patch_embed,
                PosEmbed(
                    model.cls_token, model.reg_token,
                    model.pos_embed, model.pos_drop, model.no_embed_class),
                model.patch_drop,
                model.norm_pre),
            nn.Sequential(*model.blocks[:num_blocks//4]),
            nn.Sequential(*model.blocks[num_blocks//4:2*num_blocks//4]),
            nn.Sequential(*model.blocks[2*num_blocks//4:3*num_blocks//4]),
            nn.Sequential(
                *model.blocks[3*num_blocks//4:-1],
                nn.Sequential(model.blocks[-1], model.norm))),
        nn.Sequential(
            nn.Sequential(
                PoolToken(), model.fc_norm, model.head_drop, model.head)))


def init_deit(
        rank: int,
        init_dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        init_x_batch: Tensor,
        *, model_seed: Optional[int], num_classes: Optional[int],
        model_type: str,
        patch_size: Optional[tuple[int, int]],
        model_load_params: dict[str, Any]
        ) -> nn.Module:
    """Initialize DeiT"""
    my.utils.set_torch_seed(model_seed)
    model = timm.create_model(model_type, **model_load_params)
    if patch_size is not None:
        model.set_input_size(init_x_batch.shape[2:], patch_size)
    if num_classes is not None:
        model.reset_classifier(num_classes)
    model = deit_to_sequential(model)
    return model


def deit_to_sequential8(model: VisionTransformer) -> nn.Module:
    """
    Transform timm DeiT to a layer-wise format
    """
    num_blocks = len(model.blocks)
    return nn.Sequential(
        nn.Sequential(
            nn.Sequential(
                model.patch_embed,
                PosEmbed(
                    model.cls_token, model.reg_token,
                    model.pos_embed, model.pos_drop, model.no_embed_class),
                model.patch_drop,
                model.norm_pre),
            nn.Sequential(*model.blocks[:num_blocks//6]),
            nn.Sequential(*model.blocks[num_blocks//6:2*num_blocks//6]),
            nn.Sequential(*model.blocks[2*num_blocks//6:3*num_blocks//6]),
            nn.Sequential(*model.blocks[3*num_blocks//6:4*num_blocks//6]),
            nn.Sequential(*model.blocks[4*num_blocks//6:5*num_blocks//6]),
            nn.Sequential(
                *model.blocks[5*num_blocks//4:-1],
                nn.Sequential(model.blocks[-1], model.norm))),
        nn.Sequential(
            nn.Sequential(
                PoolToken(), model.fc_norm, model.head_drop, model.head)))


def init_deit8(
        rank: int,
        init_dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        init_x_batch: Tensor,
        *, model_seed: Optional[int], num_classes: Optional[int],
        model_type: str,
        patch_size: Optional[tuple[int, int]],
        model_load_params: dict[str, Any]
        ) -> nn.Module:
    """Initialize DeiT"""
    my.utils.set_torch_seed(model_seed)
    model = timm.create_model(model_type, **model_load_params)
    if patch_size is not None:
        model.set_input_size(init_x_batch.shape[2:], patch_size)
    if num_classes is not None:
        model.reset_classifier(num_classes)
    model = deit_to_sequential8(model)
    return model


def convnext_to_sequential(model: ConvNeXt) -> nn.Module:
    """
    Transform torchvision ConvNeXt to a layer-wise format
    """
    return nn.Sequential(
        nn.Sequential(
            nn.Sequential(*model.features[0]),
            nn.Sequential(
                *model.features[1][:-1],
                nn.Sequential(model.features[1][-1], model.features[2][0])),
            *(
                nn.Sequential(
                    model.features[i-1][-1],
                    *model.features[i][:-1],
                    nn.Sequential(model.features[i][-1], model.features[i+1][0]))
                for i in range(3, len(model.features)-2, 2)),
            nn.Sequential(
                model.features[-2][-1],
                *model.features[-1][:-1],
                nn.Sequential(model.features[-1][-1], model.avgpool, model.classifier[0]))),
        nn.Sequential(
            nn.Sequential(*model.classifier[1:])))


def init_convnext(
        rank: int,
        init_dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        init_x_batch: Tensor,
        *, num_classes: Optional[int],
        model_seed: Optional[int],
        model_type: Callable[..., ConvNeXt],
        model_load_params: dict[str, Any]
        ) -> nn.Module:
    """Initialize ConvNeXt"""
    my.utils.set_torch_seed(model_seed)
    model = model_type(**model_load_params)
    if num_classes is not None:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model = convnext_to_sequential(model)
    return model


def load_model(
        rank: int,
        init_dataloader: DataLoader,
        gpu_x_transform: my.data.GPUTransform,
        gpu_y_transform: my.data.GPUTransform,
        init_x_batch: Tensor,
        *, model: nn.Module) -> nn.Module:
    """Initialize the model with the model"""
    return model


def no_splits(model: nn.Module, *, clean: bool) -> tuple[
    list[nn.Module], list[str],
    list[nn.Module], list[nn.Module], list[nn.Module]]:
    """Initialize a model"""
    models = [model]
    empty_layers = nn.ModuleList([])
    c_blocks = [empty_layers]
    s_blocks = [models[-1]]
    if clean:
        c_blocks, s_blocks = s_blocks, c_blocks
    f_blocks = [empty_layers]
    a_name = "clean" if clean else "skewed"
    model_names = [a_name]
    return models, model_names, c_blocks, s_blocks, f_blocks


def head_splits(model: nn.Module, *, clean: bool) -> tuple[
    list[nn.Module], list[str],
    list[nn.Module], list[nn.Module], list[nn.Module]]:
    """
    Initialize a model and heads
    """
    num_heads = sum(len(modality) for modality in model) - 1
    models = [copy.deepcopy(model) for _ in range(num_heads)] + [model]
    anchor = models[-1]
    curr_mod = 0
    curr_layer = 0
    for model in models[:-1]:
        for i in range(curr_mod):
            model[i] = anchor[i]
        for i in range(curr_layer+1):
            model[curr_mod][i] = anchor[curr_mod][i]
        curr_layer += 1
        if len(model[curr_mod]) <= curr_layer:
            curr_layer = 0
            curr_mod += 1
    all_layers = list(map(
        lambda model: sum(
            (nn.ModuleList(modality) for modality in model),
            nn.ModuleList([])),
        models))
    empty_layers = nn.ModuleList([])
    c_blocks =\
        [layers[i+1:] for i, layers in enumerate(all_layers[:-1])]\
        + [empty_layers]
    s_blocks = [empty_layers] * num_heads + [all_layers[-1]]
    if clean:
        c_blocks, s_blocks = s_blocks, c_blocks
    f_blocks = [layers[:i+1] for i, layers in enumerate(all_layers[:-1])]\
        + [empty_layers]
    a_name = "clean" if clean else "skewed"
    model_names =\
        [f"{a_name[0]}_{i+1}:{num_heads+1}" for i in range(num_heads)]\
        + [a_name]
    return models, model_names, c_blocks, s_blocks, f_blocks


def not_layer_splits(model: nn.Module, *, clean: bool) -> tuple[
    list[nn.Module], list[str],
    list[nn.Module], list[nn.Module], list[nn.Module]]:
    """
    Initialize a model and complements to individual layers
    """
    num_layers = sum(len(modality) for modality in model)
    models = [copy.deepcopy(model) for _ in range(num_layers)] + [model]
    anchor = models[-1]
    curr_mod = 0
    curr_layer = 0
    for model in models[:-1]:
        model[curr_mod][curr_layer] = anchor[curr_mod][curr_layer]
        curr_layer += 1
        if len(model[curr_mod]) <= curr_layer:
            curr_layer = 0
            curr_mod += 1
    all_layers = list(map(
        lambda model: sum(
            (nn.ModuleList(modality) for modality in model),
            nn.ModuleList([])),
        models))
    empty_layers = nn.ModuleList([])
    c_blocks =\
        [layers[:i]+layers[i+1:] for i, layers in enumerate(all_layers[:-1])]\
        + [empty_layers]
    s_blocks = [empty_layers] * num_layers + [all_layers[-1]]
    if clean:
        c_blocks, s_blocks = s_blocks, c_blocks
    f_blocks = [layers[i] for i, layers in enumerate(all_layers[:-1])]\
        + [empty_layers]
    a_name = "clean" if clean else "skewed"
    model_names = [f"{a_name[0]}_-{i}" for i in range(num_layers)] + [a_name]
    return models, model_names, c_blocks, s_blocks, f_blocks


def all_splits(model: nn.Module, *, clean: bool) -> tuple[
    list[nn.Module], list[str],
    list[nn.Module], list[nn.Module], list[nn.Module]]:
    """
    Initialize a model, heads, and complements to individual layers
    """
    h_obj = not_layer_splits(model, clean=clean)
    l_obj = head_splits(model, clean=clean)
    return tuple(x[:-1] + y for x, y in zip(h_obj, l_obj))


def exp_splits(model: nn.Module, *, clean: bool) -> tuple[
    list[nn.Module], list[str],
    list[nn.Module], list[nn.Module], list[nn.Module]]:
    """
    Initialize several models for optimization experiments
    """
    num_layers = sum(len(modality) for modality in model)
    models = [copy.deepcopy(model) for _ in range(num_layers + 1)] + [model]
    all_layers = list(map(
        lambda model: sum(
            (nn.ModuleList(modality) for modality in model),
            nn.ModuleList([])),
        models))
    empty_layers = nn.ModuleList([])
    c_blocks = [empty_layers] * (num_layers + 1) + [all_layers[-1]]
    s_blocks = all_layers[:-1] + [empty_layers]
    if clean:
        c_blocks, s_blocks = s_blocks, c_blocks
    f_blocks = [empty_layers] * (num_layers + 2)
    a_name = "clean" if clean else "skewed"
    b_name = "clean" if not clean else "skewed"
    model_names\
        = [f"{a_name[0]}_{i}" for i in range(num_layers)]\
            + [a_name, b_name]
    return models, model_names, c_blocks, s_blocks, f_blocks


def freeze_splits(model: nn.Module, *, clean: bool) -> tuple[
    list[nn.Module], list[str],
    list[nn.Module], list[nn.Module], list[nn.Module]]:
    """
    Initialize mostly freezed models
    """
    num_layers = sum(len(modality) for modality in model)
    models = [copy.deepcopy(model) for _ in range(num_layers)] + [model]
    all_layers = list(map(
        lambda model: sum(
            (nn.ModuleList(modality) for modality in model),
            nn.ModuleList([])),
        models))
    empty_layers = nn.ModuleList([])
    c_blocks = [empty_layers] * (num_layers + 1)
    s_blocks = [all_layers[i][i] for i in range(num_layers)] + [all_layers[-1]]
    if clean:
        c_blocks, s_blocks = s_blocks, c_blocks
    f_blocks = [all_layers[i][:i] + all_layers[i][i+1:] for i in range(num_layers)] + [empty_layers]
    a_name = "clean" if clean else "skewed"
    model_names\
        = [f"{a_name[0]}_{i}" for i in range(num_layers)] + [a_name]
    return models, model_names, c_blocks, s_blocks, f_blocks


def last_layer(model: nn.Module, *, clean: bool) -> tuple[
    list[nn.Module], list[str],
    list[nn.Module], list[nn.Module], list[nn.Module]]:
    """
    Initialize mostly freezed models
    """
    models = [copy.deepcopy(model)]
    all_layers = list(map(
        lambda model: sum(
            (nn.ModuleList(modality) for modality in model),
            nn.ModuleList([])),
        models))
    empty_layers = nn.ModuleList([])
    c_blocks = [empty_layers]
    s_blocks = [all_layers[0][-1]]
    if clean:
        c_blocks, s_blocks = s_blocks, c_blocks
    f_blocks = [all_layers[0][:-1]]
    a_name = "clean" if clean else "skewed"
    model_names = [f"{a_name[0]}_{-1}"]
    return models, model_names, c_blocks, s_blocks, f_blocks


def curr_and_last_layers(model: nn.Module, *, clean: bool) -> tuple[
    list[nn.Module], list[str],
    list[nn.Module], list[nn.Module], list[nn.Module]]:
    """
    Initialize mostly freezed models
    """
    num_layers = sum(len(modality) for modality in model)
    models = [copy.deepcopy(model) for _ in range(num_layers-1)] + [model]
    all_layers = list(map(
        lambda model: sum(
            (nn.ModuleList(modality) for modality in model),
            nn.ModuleList([])),
        models))
    empty_layers = nn.ModuleList([])
    c_blocks = [empty_layers] * num_layers
    s_blocks = [all_layers[i][i] + all_layers[i][-1] for i in range(num_layers-1)] + [all_layers[-1]]
    if clean:
        c_blocks, s_blocks = s_blocks, c_blocks
    f_blocks = [all_layers[i][:i] + all_layers[i][i+1:-1] for i in range(num_layers-1)] + [empty_layers]
    a_name = "clean" if clean else "skewed"
    model_names\
        = [f"{a_name[0]}_{i},{num_layers}" for i in range(num_layers-1)] + [a_name]
    return models, model_names, c_blocks, s_blocks, f_blocks

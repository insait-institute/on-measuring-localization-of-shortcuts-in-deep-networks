from torch import nn
import torchvision.models as tvm


def fcnn(in_dim: int, hid_dims: list[int], num_classes: int) -> nn.Module:
    """Fully-connected network"""
    return nn.Sequential(nn.Sequential(
        nn.Sequential(nn.Linear(in_dim, hid_dims[0]), nn.ReLU()),
        *(
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for in_dim, out_dim in zip(hid_dims, hid_dims[1:])),
        nn.Linear(hid_dims[-1], num_classes)))


def lenet5(num_classes: int) -> nn.Module:
    """LeNet5"""
    return nn.Sequential(
        nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 6, 5), nn.ReLU(), nn.AvgPool2d(2, 2)),
            nn.Sequential(
                nn.Conv2d(6, 16, 5), nn.ReLU(), nn.AdaptiveAvgPool2d((5, 5)))),
        nn.Sequential(
            nn.Sequential(nn.Flatten(), nn.Linear(400, 120), nn.ReLU()),
            nn.Sequential(nn.Linear(120, 84), nn.ReLU()),
            nn.Linear(84, num_classes)))


def resnet10(num_classes: int) -> tvm.ResNet:
    """ResNet-10"""
    return tvm.ResNet(
        tvm.resnet.BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

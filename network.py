import torch.nn as nn
from DiverseBranchBlock.resnet import BasicBlock, Bottleneck, ResNet

def create_resnet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, width_multiplier=1)

def create_resnet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, width_multiplier=1)

def create_resnet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, width_multiplier=1)

def create_resnet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, width_multiplier=1)

def create_resnet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, width_multiplier=1)
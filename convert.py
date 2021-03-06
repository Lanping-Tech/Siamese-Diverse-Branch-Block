import argparse
import os
import importlib
import torch
from DiverseBranchBlock.convnet_utils import switch_conv_bn_impl, switch_deploy_flag, build_model

from torchsummary import summary

parser = argparse.ArgumentParser(description='DBB Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to the weights file')
parser.add_argument('--backbone', type=str, default='resnet18', help='the name of backbone')
parser.add_argument('--n_classes', default=4, type=int, help='number of classes')

def convert():
    args = parser.parse_args()

    switch_conv_bn_impl('DBB')
    switch_deploy_flag(False)
    train_model = getattr(importlib.import_module('network'),'create_' + args.backbone)(args.n_classes)

    if 'hdf5' in args.load:
        from utils import model_load_hdf5
        model_load_hdf5(train_model, args.load)
    elif os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    for m in train_model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()

    torch.save(train_model.state_dict(), args.save)


if __name__ == '__main__':
    convert()
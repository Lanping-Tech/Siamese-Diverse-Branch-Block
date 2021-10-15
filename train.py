import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import importlib
import random
import warnings
import argparse
from tqdm import tqdm

import numpy
import torch
import torch.backends.cudnn as cudnn
from torch.nn.functional import cross_entropy, pairwise_distance, triplet_margin_loss

from dataset import load_train_data, sampling_strategy
from test import test
from utils import performance_display
from DiverseBranchBlock.convnet_utils import switch_deploy_flag, switch_conv_bn_impl, build_model

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Siamese Diverse Branch Block'
    )
    
    parser.add_argument('--backbone', type=str, default='resnet18', help='the name of backbone')
    parser.add_argument('--n_classes', default=4, type=int, help='number of classes')
    parser.add_argument('--blocktype', metavar='BLK', default='DBB', choices=['DBB', 'ACB', 'base'])

    parser.add_argument('--data_path', type=str, default='Dataset', help='the path of dataset')
    parser.add_argument('--crop_shape', default=256, type=int, help='crop shape')
    parser.add_argument('--split_rate', default=0.9, type=float, help='crop shape')
    parser.add_argument('--train_batch_size', default=100, type=int, help='training batch size')
    parser.add_argument('--val_batch_size', default=100, type=int, help='val_data batch size')

    parser.add_argument('--epochs', default=1, type=int, help='number of epochs tp train for')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lambda_', default=0.5, type=float, help='loss func weight')
    parser.add_argument('--mu_', default=10, type=float, help='loss margin')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')

    parser.add_argument('--output_path', default="output", type=str, help='output path')
    parser.add_argument('--seed', type=int, default=1, help='Random seed, a int number')
    
    return parser.parse_args()

def train(model, train_dataset, train_target_dict, val_dataset, val_target_dict, args):

    train_data_loader = torch.utils.data.DataLoader(train_dataset, args.train_batch_size, shuffle=True)
    # val_data_loader = torch.utils.data.DataLoader(valid_dataset, args.val_batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_accs, train_losses, val_accs, val_losses = [],[],[],[]

    for epoch in range(args.epochs):
        model = model.train()
        pbar = tqdm(train_data_loader)
        pbar.set_description("Epoch {}:".format(epoch))
        for data, target in pbar:
            batch_loss = 0

            pos_samples, neg_samples, neg_targets = sampling_strategy(data, target, train_dataset, train_target_dict)

            data = data.double().to(args.device)
            target = target.to(args.device)
            pos_samples = torch.from_numpy(pos_samples).double().to(args.device)
            neg_samples = torch.from_numpy(neg_samples).permute(1, 0, 2, 3, 4).double().to(args.device)
            neg_targets = torch.from_numpy(neg_targets).permute(1, 0).to(args.device)

            optimizer.zero_grad()
            ref_embedding = model(data)
            pos_embedding = model(pos_samples)

            loss_1 = cross_entropy(ref_embedding, target)
            loss_2 = cross_entropy(pos_embedding, target)
            loss_3 = 0.5 *  torch.mean(pairwise_distance(ref_embedding, pos_embedding))
            loss = (1 - args.lambda_) * loss_3 + 0.5 * args.lambda_ * (loss_1 + loss_2)
            
            batch_loss = batch_loss + loss
            loss.backward(retain_graph=True)
            loss = 0
            del pos_embedding
            torch.cuda.empty_cache()

            for i in range(len(train_target_dict.keys())-1):
                neg_embedding = model(neg_samples[i])
                neg_target = neg_targets[i]
                
                loss_1 = cross_entropy(ref_embedding, target)
                loss_2 = cross_entropy(neg_embedding, neg_target)
                loss_3 = 0.5 *  torch.clamp(args.mu_ - torch.mean(pairwise_distance(ref_embedding, neg_embedding)), min=0.0)
                loss = (1 - args.lambda_) * loss_3 + 0.5 * args.lambda_ * (loss_1 + loss_2)

                batch_loss = batch_loss + loss
                loss.backward(retain_graph=True)
                loss = 0
                del neg_embedding
                torch.cuda.empty_cache()

            optimizer.step()

            batch_loss = batch_loss / len(train_target_dict.keys())
            pbar.set_postfix(loss=batch_loss.item())
        
        # val phase
        train_acc, train_loss = test(model, train_dataset, args.train_batch_size, args.device)
        val_acc, val_loss = test(model, val_dataset, args.val_batch_size, args.device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # model save
        torch.save(model.state_dict(), args.output_path+'/epoch_{0}_train_loss_{1:>0.5}_val_loss_{2:>0.5}.pth'.format(epoch,train_loss,val_loss))

    acc_plot = {}
    acc_plot['train'] = train_accs
    acc_plot['val'] = val_accs
    loss_plot = {}
    loss_plot['train'] = train_losses
    loss_plot['val'] = val_losses

    performance_display(acc_plot, 'Accuracy', args.output_path)
    performance_display(loss_plot, 'Loss', args.output_path)


if __name__ == '__main__':
    args = parse_arguments()
    seed_torch(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load train and val data
    train_dataset, train_target_dict, val_dataset, val_target_dict = load_train_data(args.data_path, crop_shape=32, split_rate=0.9)

    # build model
    switch_deploy_flag(False)
    switch_conv_bn_impl(args.blocktype)
    model = getattr(importlib.import_module('network'),'create_' + args.backbone)(args.n_classes)
    model = model.double().to(args.device)

    # train phase
    train(model, train_dataset, train_target_dict, val_dataset, val_target_dict, args)



import os

import random
import numpy
import argparse
import importlib

from sklearn import metrics
import torch
import torch.backends.cudnn as cudnn
from torch.nn.functional import cross_entropy, softmax
from dataset import load_test_data

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

    parser.add_argument('--data_path', type=str, default='Dataset', help='the path of dataset')
    parser.add_argument('--crop_shape', default=256, type=int, help='crop shape')
    parser.add_argument('--split_rate', default=0.9, type=float, help='crop shape')
    parser.add_argument('--test_batch_size', default=100, type=int, help='testing batch size')

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')

    parser.add_argument('--output_path', default="output", type=str, help='output path')
    parser.add_argument('--ck_path', default="****.ckpt", type=str, help='check point path')
    parser.add_argument('--seed', type=int, default=1, help='Random seed, a int number')
    
    return parser.parse_args()

def test(model, dataset, batch_size, device, test=False):

    model = model.eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    total_loss = 0
    predict_all = numpy.array([], dtype=int)
    labels_all = numpy.array([], dtype=int)
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            data, target = batch
            data = data.double().to(device)
            target = target.to(device)

            y_pred = model(data)

            loss = cross_entropy(y_pred, target)
            total_loss += loss

            target = target.data.cpu().numpy()
            predict = torch.max(softmax(y_pred).data, 1)[1].cpu().numpy()
            labels_all = numpy.append(labels_all, target)
            predict_all = numpy.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, total_loss / len(data_loader), report, confusion
    return acc, total_loss / len(data_loader)


if __name__ == '__main__':
    args = parse_arguments()
    seed_torch(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load train and val data
    test_dataset = load_test_data(args.data_path, crop_shape=32)

    # build model
    model = getattr(importlib.import_module('network'),'create_' + args.backbone)(args.n_classes)
    model.load_state_dict(torch.load(os.path.join(args.out_path,args.ck_path)))
    model = model.double().to(args.device)
    

    # test phase
    test_acc, test_loss, report, confusion = test(model, test_dataset, args.test_batch_size, args.device, test=True)

    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)
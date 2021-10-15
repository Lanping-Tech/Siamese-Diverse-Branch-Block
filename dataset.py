import os
import random
import numpy
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

def list2dict(labels):
    label2index = {}
    for i, label in enumerate(labels):
        if label in label2index:
            label2index[label].append(i)
        else:
            label2index[label] = []
            label2index[label].append(i)
    return label2index

def sampling_strategy(batch, target, dataset, target_dict):
    b, c, h, w = batch.shape
    pos_samples = numpy.empty((0, c, h, w))
    neg_samples = numpy.empty((0, len(target_dict)-1, c, h, w))
    neg_targets = []
    for i in range(b):
        pos_indices = random.choice(target_dict[int(target[i].cpu().numpy())])
        pos_sample = numpy.expand_dims(dataset.__getitem__(pos_indices)[0], axis=0)
        pos_samples = numpy.concatenate((pos_samples, pos_sample), axis=0)

        other_labels = set(target_dict.keys())-{int(target[i].cpu().numpy())}
        neg_sub_samples = numpy.empty((0, c, h, w))
        neg_sub_targets = []
        for label in other_labels:
            neg_sub_targets.append(label)
            neg_indices = random.choice(target_dict[label])
            neg_sample = numpy.expand_dims(dataset.__getitem__(neg_indices)[0], axis=0)
            neg_sub_samples = numpy.concatenate((neg_sub_samples, neg_sample), axis=0)
        
        neg_samples = numpy.concatenate((neg_samples, numpy.expand_dims(neg_sub_samples, axis=0)), axis=0)
        neg_targets.append(neg_sub_targets)

    return pos_samples, neg_samples, numpy.array(neg_targets)

def load_data(data_path, crop_shape=32, is_train=True):
    datadir = os.path.join(data_path, 'train') if is_train else os.path.join(data_path, 'test')

    data_trainsforms = transforms.Compose([transforms.Resize((crop_shape,crop_shape)),
                                            transforms.ToTensor(),])
 
    dataset = datasets.ImageFolder(datadir,transform=data_trainsforms)
    return dataset

def load_train_data(data_path, crop_shape=32, split_rate=0.9):
    total_dataset = load_data(data_path, crop_shape, True)
    total_target_list = numpy.array([s[1] for s in total_dataset.samples])

    total_count = len(total_dataset.samples)
    train_count = int(split_rate * total_count)
    val_count = total_count - train_count
    train_dataset, valid_dataset = random_split(total_dataset, (train_count, val_count))

    train_target_list = total_target_list[train_dataset.indices]
    train_target_dict = list2dict(train_target_list)

    val_target_list = total_target_list[valid_dataset.indices]
    val_target_dict = list2dict(val_target_list)

    return train_dataset, train_target_dict, valid_dataset, val_target_dict

def load_test_data(data_path, crop_shape=32):
    test_dataset = load_data(data_path, crop_shape, False)
    return test_dataset


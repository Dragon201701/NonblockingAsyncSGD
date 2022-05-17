import os
from random import Random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from torchvision.datasets import CIFAR10


"""
This file is used to Partition data and allocate them to different workers.
The code is reused from torch official document. Reference： https://pytorch.org/tutorials/intermediate/dist_tuto.html
Dataset Partition helper is from official document.
"""


""" Partitioning MNIST """
def partition_dataset(args):
    CIFAR10transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    dataset = CIFAR10('./data', train=True, download=True,
                             transform=CIFAR10transform)

    size = dist.get_world_size() - 1
    bsz = args.batch_size / float(size)
    bsz = int(bsz)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank() - 1)
    #train_set = DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=args.num_workers)
    train_set = DataLoader(partition, batch_size=bsz, shuffle=True)
    return train_set, bsz



""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


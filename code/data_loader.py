import os
from random import Random
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist


def get_data_loader(args):
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = KaggleAmazonDataset(args.data, data_transforms)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return train_loader


def partition_dataset(args):
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = KaggleAmazonDataset(args.data, data_transforms)
    size = dist.get_world_size()
    partition_sizes = [1.0 / (size - 1) for _ in range(1, size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank()-1)
    train_set = DataLoader(partition, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return train_set


class KaggleAmazonDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        train_csv = os.path.join(root_dir, 'train.csv')
        train_df = pd.read_csv(train_csv)
        self.x_train = train_df['image_name']
        self.y_train = train_df['tags']

    def __len__(self):
        return len(self.x_train.index)

    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'train-jpg',
                                self.x_train[idx]+'.jpg')
        image = self._load_image(img_name)
        labels = [0]*17
        positive = self.y_train[idx].split()
        positive = [int(s) for s in positive]

        labels = torch.zeros(17)
        labels[positive] = 1

        if self.transform:
            image = self.transform(image)

        return image, labels


class Partition(object):
    """ Dataset partitioning helper """

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

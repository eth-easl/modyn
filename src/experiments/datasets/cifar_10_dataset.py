from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from torchvision import transforms
import torch

from .task_dataset import TaskDataset

def get_cifar10_dataset(train_aug=None, val_aug=None, version='normal', configs=None):

    def normalize_dataset(data):
        mean = data.mean(axis=(0,1,2)) / 255.0
        std = data.std(axis=(0,1,2)) / 255.0
        normalize = transforms.Normalize(mean=mean, std=std)
        return normalize

    def load_cifar10_data(filename):
        with open('./data/cifar10/cifar-10-batches-py/'+ filename, 'rb') as file:
            batch = pickle.load(file, encoding='latin1')

        features = batch['data']
        labels = batch['labels']
        return features, labels

    # Load files
    batch_1, labels_1 = load_cifar10_data('data_batch_1')
    batch_2, labels_2 = load_cifar10_data('data_batch_2')
    batch_3, labels_3 = load_cifar10_data('data_batch_3')
    batch_4, labels_4 = load_cifar10_data('data_batch_4')
    batch_5, labels_5 = load_cifar10_data('data_batch_5')
    testX, testY = load_cifar10_data('test_batch')

    trainX = np.concatenate([batch_1,batch_2,batch_3,batch_4,batch_5], 0)
    trainY = torch.from_numpy(np.concatenate([labels_1,labels_2,labels_3,labels_4,labels_5], 0))

    trainX = np.reshape(trainX, (-1, 32, 32, 3))
    testX = np.reshape(testX, (-1, 32, 32, 3)) 

    testY = torch.from_numpy(np.array(testY))

    classes = ('airplane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if train_aug is None:
        train_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((40, 40)),      
            transforms.RandomCrop((32, 32)),   
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize_dataset(trainX) 
        ])

    if val_aug is None:
        val_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize_dataset(testX)
        ])

    val_configs = configs.copy()
    val_configs['blurry'] = 0

    if version == 'normal':
        return {
            'train': CIFAR10Dataset(trainX, trainY, train_aug),
            'test': CIFAR10Dataset(testX, testY, val_aug),
        }
    elif version == 'split':
        return {
            'train': SplitCIFAR10Dataset(trainX, trainY, train_aug, configs),
            'test': SplitCIFAR10Dataset(testX, testY, val_aug, val_configs),
        }
    else:
        raise NotImplementedError()


class CIFAR10Dataset(Dataset):
    def __init__(self, x, y, augmentation):
        self.x = x
        self.y = y
        self.augmentation = augmentation 
    
    def __len__(self):
        return len(self.x)
    
    def is_task_based(self):
        return False

    def __getitem__(self, idx):
        if self.augmentation is not None:
            return self.augmentation(self.x[idx]), self.y[idx]
        elif self.y is None:
            return [self.x[idx]]
        return self.x[idx], self.y[idx]

    def __repr__(self):
        return f'CIFAR10-Normal'


class SplitCIFAR10Dataset(TaskDataset):
    def __init__(self, x, y, augmentation, configs):
        super(SplitCIFAR10Dataset, self).__init__(['Airplane/Car', 'Bird/Cat', 'Deer/Dog', 'Frog/Horse', 'Ship/Truck'], configs)
        self.augmentation = augmentation

        task_idx, self.x, self.y = [], [], []
        task_idx.append((y==0) | (y==1))
        task_idx.append((y==2) | (y==3))
        task_idx.append((y==4) | (y==5))
        task_idx.append((y==6) | (y==7))
        task_idx.append((y==8) | (y==9))
        self.full_x = x
        self.full_y = y
        self.full_mode = False

        for idx, task in enumerate(task_idx):
            self.x.append(x[task])
            if configs['collapse_targets']: 
                self.y.append(y[task] - 2*idx)
            else:
                self.y.append(y[task])

    def __repr__(self):
        return f'CIL-CIFAR10-Blurry{self.blurry}'

    def __len__(self):
        if self.full_mode:
            return len(self.full_x)
        else:
            return int(len(self.x[self.active_idx]) * (1+self.blurry/100))

    def __getitem__(self, idx):
        if self.full_mode:
            x = self.full_x[idx]
            y = self.full_y[idx]
        elif idx > len(self.x[self.active_idx]):
            minor_idx = idx - len(self.x[self.active_idx])
            minor_task = minor_idx % (len(self.tasks) - 1)
            minor_idx = minor_idx // (len(self.tasks) - 1)
            x = self.x[minor_task][minor_idx]
            y = self.y[minor_task][minor_idx]
        else:
            x = self.x[self.active_idx][idx]
            y = self.y[self.active_idx][idx]

        if self.augmentation is not None:
            return self.augmentation(x), y
        elif y is None:
            return [x]
        return x, y


if __name__ == '__main__':
    A = get_cifar10_dataset()
    print(A['train'][0][0].shape)
    print(A['test'][0][0].shape)
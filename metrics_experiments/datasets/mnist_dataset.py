from torch.utils.data import DataLoader, Dataset
from keras.datasets import mnist
from torchvision import transforms
import torch
from .task_dataset import TaskDataset

def get_mnist_dataset(train_aug=None, val_aug=None, version='normal', collapse_targets=True):
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = torch.from_numpy((trainX/255).reshape(-1, 1, 28, 28)).float()
    testX = torch.from_numpy((testX/255).reshape(-1, 1, 28, 28)).float()
    trainY = torch.from_numpy(trainY).long()
    testY = torch.from_numpy(testY).long()

    if train_aug is None:
        train_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees = 30),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            # TODO missing the normalize
        ])

    if val_aug is None: 
        val_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # TODO not actually sure what the honte is here
        ])

    if version == 'normal':
        return {
            'train': MNISTDataset(trainX, trainY, train_aug),
            'test': MNISTDataset(testX, testY, val_aug),
        }
    elif version == 'split':
        return {
            'train': SplitMNISTDataset(trainX, trainY, train_aug, collapse_targets),
            'test': SplitMNISTDataset(testX, testY, val_aug, collapse_targets),
        }
    else:
        raise NotImplementedError()

class MNISTDataset(Dataset):
    def __init__(self, x, y, augmentation):
        self.x = x
        self.y = y
        self.augmentation = augmentation

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.augmentation is not None:
            return self.augmentation(self.x[idx]), self.y[idx]
        elif self.y is None:
            return [self.x[idx]]
        return self.x[idx], self.y[idx]

    def is_task_based(self):
        return False

class SplitMNISTDataset(TaskDataset):
    def __init__(self, x, y, augmentation, collapse_targets):
        super(SplitMNISTDataset, self).__init__(['0/1', '2/3', '4/5', '6/7', '8/9'])
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
            if collapse_targets: 
                self.y.append(y[task] - 2*idx)
            else:
                self.y.append(y[task])

    
    def __len__(self):
        return len(self.full_x) if self.full_mode else len(self.x[self.active_idx]) 

    def __getitem__(self, idx):
        x = self.full_x[idx] if self.full_mode else self.x[self.active_idx][idx]
        y = self.full_y[idx] if self.full_mode else self.y[self.active_idx][idx]

        if self.augmentation is not None:
            return self.augmentation(x), y
        elif y is None:
            return [x]
        return x, y
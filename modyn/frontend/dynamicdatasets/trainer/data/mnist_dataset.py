from typing import Optional

import torch.nn.Module
import torch.Tesnor
from torch.utils.data import Dataset
from torchvision import datasets as dts
from torchvision import transforms


class MNISTDataset(Dataset):
    # TODO(roxanastiuca): please check the typing in this class. This class is never used currently.

    def __init__(self, x: list, y: list, augmentation: torch.nn.Module):
        self.x = x
        self.y = y
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.augmentation is not None:
            return self.augmentation(self.x[idx]), self.y[idx]
        elif self.y is None:
            return [self.x[idx]]
        return self.x[idx], self.y[idx]

    def is_task_based(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "MNIST-Normal"


def get_mnist_dataset(
    train_aug: Optional[torch.nn.Module] = None,
    val_aug: Optional[torch.nn.Module] = None,
    version: str = "normal",
) -> dict:
    if train_aug is None:
        train_aug = transforms.Compose(
            [
                transforms.RandomAffine(degrees=30),
                transforms.RandomPerspective(),
                transforms.ToTensor(),
            ]
        )

    if val_aug is None:
        val_aug = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    traindt = dts.MNIST(
        root="data",
        train=True,
        transform=train_aug,
        download=True,
    )
    testdt = dts.MNIST(root="data", train=False, transform=val_aug)

    if version == "normal":
        return {
            "train": traindt,
            "test": testdt,
        }
    # elif version == 'split':
    #     return {
    #         'train': SplitMNISTDataset(trainX, trainY, train_aug, configs),
    #         'test': SplitMNISTDataset(testX, testY, val_aug, val_configs),
    #     }
    else:
        raise NotImplementedError()

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch


class BufferDataset(Dataset):
    def __init__(self, x, y, augmentation, fake_size=0):
        self.x = x
        self.y = y
        self.augmentation = augmentation
        self.fake_size = fake_size

    def __len__(self):
        if self.fake_size > 0 and len(self.x) > 0:
            return self.fake_size
        return len(self.x)

    def __getitem__(self, idx):
        if self.fake_size > 0 and idx >= len(self.x):
            idx %= len(self.x)

        if self.augmentation is not None:
            return self.augmentation(self.x[idx]), self.y[idx]
        elif self.y is None:
            return [self.x[idx]]
        return self.x[idx], self.y[idx]

    def update(self, x, y):
        self.x = x
        self.y = y

    def update(self, buffer):
        self.x = torch.cat(buffer.bufferX[:buffer.get_size()])
        self.y = buffer.bufferY[:buffer.get_size()]
        self.weights = buffer.weights[:buffer.get_size()]

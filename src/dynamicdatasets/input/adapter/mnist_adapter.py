import json

from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader

from . import BaseAdapter


class MNISTAdapter(BaseAdapter):
    """MNISTAdapter is an adapter for the MNIST dataset.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        dataset = MNIST(root='./data')
        self.__dataloader = DataLoader(
            dataset,
            batch_size=self._config['input']['size'],
            shuffle=True,
            num_workers=5)

    def _get(self) -> list[bytes]:
        data: list[bytes] = []

        images, labels = next(iter(self.__dataloader))

        for i in range(len(images)):
            d = {'image': images[i], 'label': labels[i]}
            data_bytes = json.dumps(d).encode('utf-8')
            data.append(data_bytes)
        return data

import json

from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from . import BaseAdapter


class MNISTAdapter(BaseAdapter):
    """MNISTAdapter is an adapter for the MNIST dataset.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.__dataloader = DataLoader(
            MNIST(root='./data',
                  download=True,
                  transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self._config['input']['send_batch_size'],
            shuffle=True)

    def get_next(self) -> list[bytes]:
        data: list[bytes] = []

        images, labels = next(iter(self.__dataloader))

        for i in range(len(images)):
            d = {
                'image': images[i].numpy().tolist(),
                'label': labels[i].item()}
            data_json = json.dumps(d)
            data.append(data_json)
        return data

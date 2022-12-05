import json
import uuid 

from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from .base import BaseSource


class MNISTDataSource(BaseSource):
    """
    MNISTAdapter is an adapter for the MNIST dataset.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._dataloader = DataLoader(
            MNIST(root='./data',
                download=True,
                transform=transforms.Compose([transforms.ToTensor()])),
                batch_size=self._config['storage']['data_source']['batch_size'],
                shuffle=True)

    def get_next(self, limit: int) -> tuple[list[str], list[str]]:
        data: list[bytes] = []
        keys: list[str] = []

        print("Getting data from source")
        images, labels = next(iter(self._dataloader))
        print("Got data from source")

        for i in range(len(images)):
            d = {
                'image': images[i].numpy().tolist(),
                'label': labels[i].item()}
            data_json = json.dumps(d)
            data.append(data_json)
            keys.append(uuid.uuid4().hex)
        return keys, data

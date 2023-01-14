import json

import torch
from frontend.dynamicdatasets.trainer.data.online_dataset import OnlineDataset


class OnlineMNISTDataset(OnlineDataset):
    def _process(self, data: list) -> list:
        """
        Override to add custom data processing.

        Args:
            data: sequence of elements from storage, most likely as json strings

        Returns:
            sequence of processed elements
        """
        images = torch.tensor(list(map(lambda x: json.loads(x)["image"], data)), dtype=torch.float32)
        labels = torch.tensor(list(map(lambda x: json.loads(x)["label"], data)))
        train_data = list(map(lambda i: (images[i], labels[i]), range(len(images))))
        return train_data

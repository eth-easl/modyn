from torchvision import transforms

from modyn.gpu_node.data.online_dataset import OnlineDataset


class MNISTDataset(OnlineDataset):

    """
    Dataset fot MNIST.
    """

    def _process(self, data: list) -> list:

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        processed_data = [transform(sample) for sample in data]
        return processed_data

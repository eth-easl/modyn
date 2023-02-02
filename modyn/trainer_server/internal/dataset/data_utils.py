from typing import Optional

import torch
from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset


def prepare_dataloaders(
    training_id: int,
    dataset_id: str,
    num_dataloaders: int,
    batch_size: int,
    bytes_parser: str,
    transform: list[str],
    train_until_sample_id: str,
) -> tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Gets the proper dataset according to the dataset id, and creates the proper dataloaders.

    Parameters:
        training_id (int): ID of the training experiment
        dataset_id (str): ID of the dataset
        num_dataloaders (int): Number of PyTorch data workers for the dataloader
        batch_size (int): Batch size used for training
        transform (list[str]): List of serialized torchvision transforms for the samples, before loading.

    Returns:
        tuple[Optional[torch.utils.data.DataLoader]]: Dataloaders for train and validation

    """

    train_set = OnlineDataset(training_id, dataset_id, bytes_parser, transform, train_until_sample_id)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_dataloaders)

    # TODO(#50): what to do with the val set in the general case?
    val_dataloader = None
    return train_dataloader, val_dataloader

from typing import Optional

import torch
from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset


def prepare_dataloaders(
    pipeline_id: int,
    trigger_id: int,
    dataset_id: str,
    num_dataloaders: int,
    batch_size: int,
    bytes_parser: str,
    transform: list[str],
    storage_address: str,
    selector_address: str,
) -> tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Gets the proper dataset according to the dataset id, and creates the proper dataloaders.

    Parameters:
        pipeline_id (int): ID of the pipeline
        trigger_id (int): ID of the specific trigger
        dataset_id (str): ID of the dataset
        num_dataloaders (int): Number of PyTorch data workers for the dataloader
        batch_size (int): Batch size used for training
        bytes_parser (str): Serialized Python code,
            used for converting bytes to a form useful for futher transformations (such as Tensors).
        transform (list[str]): List of serialized torchvision transforms for the samples, before loading.
        storage_address (str): Address of the Storage endpoint that the OnlineDataset workers connect to.
        selector_address (str): Address of the Selector endpoint that the OnlineDataset workers connect to.
    Returns:
        tuple[Optional[torch.utils.data.DataLoader]]: Dataloaders for train and validation

    """

    train_set = OnlineDataset(
        pipeline_id, trigger_id, dataset_id, bytes_parser, transform, storage_address, selector_address
    )
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_dataloaders)

    # TODO(#50): what to do with the val set in the general case?
    val_dataloader = None
    return train_dataloader, val_dataloader

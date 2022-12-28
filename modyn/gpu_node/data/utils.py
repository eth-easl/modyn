from typing import Optional
import torch
from modyn.gpu_node.data.cifar_dataset import get_cifar_datasets
from modyn.gpu_node.data.online_dataset import OnlineDataset


def prepare_dataloaders(
    training_id: int,
    dataset_id: str,
    num_dataloaders: int,
    batch_size: int
) -> tuple[Optional[torch.utils.data.DataLoader]]:

    """
    Gets the proper dataset according to the dataset id, and creates the proper dataloaders.

    Returns:
        tuple[Optional[torch.utils.data.DataLoader]]: Dataloaders for train and validation

    """

    if dataset_id == "cifar10":
        train_set, val_set = get_cifar_datasets()

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                       shuffle=True, num_workers=num_dataloaders)

        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                     shuffle=False, num_workers=num_dataloaders)

    elif dataset_id == "online":
        train_set = OnlineDataset(training_id, None)

        # TODO(#50): what to do with the val set in the general case?
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                       num_workers=num_dataloaders)

        val_dataloader = None

    else:
        return None, None

    return train_dataloader, val_dataloader

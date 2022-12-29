from typing import Optional
import torch
from modyn.utils import dynamic_module_import


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

    dataset_module = dynamic_module_import("modyn.gpu_node.data")
    assert hasattr(dataset_module, dataset_id)

    dataset_handler = getattr(dataset_module, dataset_id)

    train_set = dataset_handler(training_id)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   num_workers=num_dataloaders)

    # TODO(#50): what to do with the val set in the general case?
    val_dataloader = None
    return train_dataloader, val_dataloader

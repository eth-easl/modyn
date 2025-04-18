import logging
import pathlib

import torch

from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset
from modyn.trainer_server.internal.dataset.per_class_online_dataset import PerClassOnlineDataset

logger = logging.getLogger(__name__)

# pylint: disable=too-many-locals


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
    training_id: int,
    num_prefetched_partitions: int,
    parallel_prefetch_requests: int,
    shuffle: bool,
    tokenizer: str | None,
    log_path: pathlib.Path | None,
    drop_last: bool = True,
    include_labels: bool = True,
    bytes_parser_target: str | None = None,
    transform_target: list[str] | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Gets the proper dataset according to the dataset id, and creates the
    proper dataloaders.

    Parameters:
        pipeline_id (int): ID of the pipeline
        trigger_id (int): ID of the specific trigger
        dataset_id (str): ID of the dataset
        num_dataloaders (int): Number of PyTorch data workers for the dataloader
        batch_size (int): Batch size used for training
        bytes_parser (str): Serialized Python code,
            used for converting bytes to a form useful for further transformations (such as Tensors).
        transform (list[str]): List of serialized torchvision transforms for the samples, before loading.
        shuffle: Whether to shuffle the order of partitions and data within partitions
        tokenizer (optional[str]): Optional tokenizer for NLP tasks
        storage_address (str): Address of the Storage endpoint that the OnlineDataset workers connect to.
        selector_address (str): Address of the Selector endpoint that the OnlineDataset workers connect to.
    Returns:
        tuple[Optional[torch.utils.data.DataLoader]]: Dataloaders for train and validation
    """
    logger.debug("Creating OnlineDataset.")
    train_set = OnlineDataset(
        pipeline_id,
        trigger_id,
        dataset_id,
        bytes_parser,
        transform,
        storage_address,
        selector_address,
        training_id,
        num_prefetched_partitions,
        parallel_prefetch_requests,
        shuffle,
        tokenizer,
        log_path,
        include_labels,
        bytes_parser_target,
        transform_target,
    )
    logger.debug("Creating DataLoader.")
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_dataloaders,
        drop_last=drop_last,
        timeout=60 if num_dataloaders > 0 else 0,
    )

    # TODO(#50): what to do with the val set in the general case?
    val_dataloader = None
    return train_dataloader, val_dataloader


def prepare_per_class_dataloader_from_online_dataset(
    online_dataset: OnlineDataset,
    batch_size: int,
    num_workers: int,
    initial_filtered_label: int,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    # TODO(#289): Replace inefficient per class dataloader
    dataset = PerClassOnlineDataset(  # type: ignore  # pylint: disable=too-many-function-args
        online_dataset._pipeline_id,
        online_dataset._trigger_id,
        online_dataset._dataset_id,
        online_dataset._bytes_parser,
        online_dataset._serialized_transforms,
        online_dataset._storage_address,
        online_dataset._selector_address,
        online_dataset._training_id,
        initial_filtered_label,
        online_dataset._num_prefetched_partitions,
        online_dataset._parallel_prefetch_requests,
        online_dataset._shuffle,
        online_dataset._tokenizer_name,
        online_dataset._include_labels,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)

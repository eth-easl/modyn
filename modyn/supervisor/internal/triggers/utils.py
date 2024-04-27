import logging
import random
from typing import Optional, Union

from modyn.supervisor.internal.triggers.model_downloader import ModelDownloader
from modyn.supervisor.internal.triggers.trigger_datasets import (
    DataLoaderInfo,
    OnlineTriggerDataset,
    FixedKeysDataset,
)

import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def prepare_trigger_dataloader_by_trigger(
    trigger_id: int,
    dataloader_info: DataLoaderInfo,
    data_points_in_trigger: Optional[int] = None,
    sample_size: Optional[int] = None,
) -> DataLoader:
    sample_prob: Optional[float] = None
    if data_points_in_trigger is not None and sample_size is not None:
        sample_prob = sample_size / data_points_in_trigger

    train_set = OnlineTriggerDataset(
        dataloader_info.pipeline_id,
        trigger_id,
        dataloader_info.dataset_id,
        dataloader_info.bytes_parser,
        dataloader_info.transform_list,
        dataloader_info.storage_address,
        dataloader_info.selector_address,
        dataloader_info.training_id,
        dataloader_info.num_prefetched_partitions,
        dataloader_info.parallel_prefetch_requests,
        dataloader_info.tokenizer,
        sample_prob,
    )

    logger.debug("Creating online trigger DataLoader.")
    return DataLoader(train_set, batch_size=dataloader_info.batch_size, num_workers=dataloader_info.num_dataloaders)


def prepare_trigger_dataloader_fixed_keys(
    trigger_id: int,
    dataloader_info: DataLoaderInfo,
    keys: list[int],
    sample_size: Optional[int] = None,
) -> DataLoader:
    if sample_size is not None:
        keys = random.sample(keys, min(len(keys), sample_size))

    train_set = FixedKeysDataset(
        dataloader_info.dataset_id,
        dataloader_info.bytes_parser,
        dataloader_info.transform_list,
        dataloader_info.storage_address,
        trigger_id,
        keys,
        dataloader_info.tokenizer,
    )

    logger.debug("Creating fixed keys DataLoader.")
    return DataLoader(train_set, batch_size=dataloader_info.batch_size, num_workers=dataloader_info.num_dataloaders)
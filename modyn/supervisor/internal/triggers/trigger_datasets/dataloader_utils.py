import logging
import random
from typing import Optional

from modyn.supervisor.internal.triggers.trigger_datasets import (
    DataLoaderInfo,
    OnlineTriggerDataset,
    TriggerDatasetGivenKeys,
)
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

    # train_set_list = []
    # for x in train_set:
    #     train_set_list.append(x)
    # all_keys = [x[0] for x in train_set_list]
    # logger.debug(f"[DataCheck][Dataloader][Reference sorted]{sorted(all_keys)}")

    logger.debug("Creating DataLoader.")
    return DataLoader(train_set, batch_size=dataloader_info.batch_size, num_workers=dataloader_info.num_dataloaders)


def prepare_trigger_dataloader_given_keys(
    trigger_id: int,
    dataloader_info: DataLoaderInfo,
    keys: list[int],
    sample_size: Optional[int] = None,
) -> DataLoader:
    if sample_size is not None:
        keys = random.sample(keys, min(len(keys), sample_size))

    train_set = TriggerDatasetGivenKeys(
        dataloader_info.dataset_id,
        dataloader_info.bytes_parser,
        dataloader_info.transform_list,
        dataloader_info.storage_address,
        trigger_id,
        keys,
        dataloader_info.tokenizer,
    )

    # train_set_list = []
    # for x in train_set:
    #     train_set_list.append(x)
    # all_keys = [x[0] for x in train_set_list]
    # logger.debug(f"[DataCheck][Dataloader][Current sorted]{sorted(all_keys)}")

    logger.debug("Creating DataLoader.")
    return DataLoader(train_set, batch_size=dataloader_info.batch_size, num_workers=dataloader_info.num_dataloaders)

import logging
import random

from torch.utils.data import DataLoader

from modyn.supervisor.internal.triggers.utils.datasets.dataloader_info import (
    DataLoaderInfo,
)
from modyn.supervisor.internal.triggers.utils.datasets.fixed_keys_dataset import (
    FixedKeysDataset,
)
from modyn.supervisor.internal.triggers.utils.datasets.online_trigger_dataset import (
    OnlineTriggerDataset,
)

logger = logging.getLogger(__name__)


def prepare_trigger_dataloader_by_trigger(
    trigger_id: int,
    dataloader_info: DataLoaderInfo,
    data_points_in_trigger: int | None = None,
    sample_size: int | None = None,
) -> DataLoader:
    sample_prob: float | None = None
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
        dataloader_info.shuffle,
        dataloader_info.tokenizer,
        sample_prob,
    )

    logger.debug("Creating online trigger DataLoader.")
    return DataLoader(
        train_set,
        batch_size=dataloader_info.batch_size,
        num_workers=dataloader_info.num_dataloaders,
    )


def prepare_trigger_dataloader_fixed_keys(
    dataloader_info: DataLoaderInfo,
    keys: list[int],
    sample_size: int | None = None,
) -> DataLoader:
    if sample_size is not None:
        keys = random.sample(keys, min(len(keys), sample_size))

    train_set = FixedKeysDataset(
        dataloader_info.dataset_id,
        dataloader_info.bytes_parser,
        dataloader_info.transform_list,
        dataloader_info.storage_address,
        keys,
        dataloader_info.bytes_parser_target,
        dataloader_info.transform_list_target,
        dataloader_info.include_labels,
        dataloader_info.tokenizer,
        dataloader_info.max_token_length,
    )

    logger.debug("Creating fixed keys DataLoader.")
    return DataLoader(
        train_set,
        batch_size=dataloader_info.batch_size,
        num_workers=dataloader_info.num_dataloaders,
    )



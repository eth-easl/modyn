import logging
import random
from typing import Optional, Union

import pandas as pd
import torch
from modyn.supervisor.internal.triggers.embedding_encoder_utils import EmbeddingEncoder
from modyn.supervisor.internal.triggers.trigger_datasets import DataLoaderInfo, FixedKeysDataset, OnlineTriggerDataset
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


def get_embeddings(embedding_encoder: EmbeddingEncoder, dataloader: DataLoader) -> torch.Tensor:
    """
    input: embedding_encoder with downloaded model
    output: embeddings Tensor
    """
    assert embedding_encoder._model is not None
    all_embeddings: Optional[torch.Tensor] = None

    embedding_encoder._model.model.eval()
    embedding_encoder._model.model.embedding_recorder.start_recording()

    with torch.no_grad():
        for batch in dataloader:
            data: Union[torch.Tensor, dict]
            if isinstance(batch[1], torch.Tensor):
                data = batch[1].to(embedding_encoder.device)
            elif isinstance(batch[1], dict):
                data: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
                for name, tensor in batch[1].items():
                    data[name] = tensor.to(embedding_encoder.device)
            else:
                raise ValueError(f"data type {type(batch[1])} not supported")

            with torch.autocast(embedding_encoder.device_type, enabled=embedding_encoder.amp):
                embedding_encoder._model.model(data)
                embeddings = embedding_encoder._model.model.embedding_recorder.embedding
                if all_embeddings is None:
                    all_embeddings = embeddings
                else:
                    all_embeddings = torch.cat((all_embeddings, embeddings), 0)

    embedding_encoder._model.model.embedding_recorder.end_recording()

    return all_embeddings


def convert_tensor_to_df(t: torch.Tensor, column_name_prefix: Optional[str] = None) -> pd.DataFrame:
    matrix_numpy = t.cpu().detach().numpy()
    df = pd.DataFrame(matrix_numpy).astype("float64")
    if column_name_prefix is not None:
        df.columns = [column_name_prefix + str(x) for x in df.columns]
    return df

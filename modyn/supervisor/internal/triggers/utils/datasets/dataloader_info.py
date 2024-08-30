# TODO(415): Unify with similar classes in trainer_server and evaluator

from dataclasses import dataclass


@dataclass
class DataLoaderInfo:
    pipeline_id: int
    dataset_id: str
    num_dataloaders: int
    batch_size: int
    bytes_parser: str
    transform_list: list[str]
    storage_address: str
    selector_address: str
    num_prefetched_partitions: int
    parallel_prefetch_requests: int
    shuffle: bool
    tokenizer: str | None = None
    training_id: int = -1

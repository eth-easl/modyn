from typing import Optional


# TODO(415): Unify with similar classes in trainer_server and evaluator
class DataLoaderInfo:
    def __init__(
        self,
        pipeline_id: int,
        dataset_id: str,
        num_dataloaders: int,
        batch_size: int,
        bytes_parser: str,
        transform_list: list[str],
        storage_address: str,
        selector_address: str,
        num_prefetched_partitions: int,
        parallel_prefetch_requests: int,
        shuffle: bool,
        tokenizer: Optional[str],
    ):
        self.pipeline_id = pipeline_id
        self.dataset_id = dataset_id
        self.num_dataloaders = num_dataloaders
        self.batch_size = batch_size
        self.bytes_parser = bytes_parser
        self.transform_list = transform_list
        self.storage_address = storage_address
        self.selector_address = selector_address
        self.num_prefetched_partitions = num_prefetched_partitions
        self.parallel_prefetch_requests = parallel_prefetch_requests
        self.tokenizer = tokenizer
        self.training_id = -1
        self.shuffle = shuffle

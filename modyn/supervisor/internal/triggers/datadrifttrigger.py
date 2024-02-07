from modyn.supervisor.internal.triggers.trigger import Trigger
from modyn.supervisor.internal.triggers.trigger_dataset import prepare_trigger_dataloader_given_keys, prepare_trigger_dataloader_by_trigger
from modyn.utils import dynamic_module_import
from modyn.common.ftp import download_trained_model

import grpc
from modyn.storage.internal.grpc.generated.storage_pb2 import (  # pylint: disable=no-name-in-module
    GetDataPerWorkerRequest,
    GetDataPerWorkerResponse,
    GetRequest,
    GetResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils.utils import (
    BYTES_PARSER_FUNC_NAME,
    grpc_common_config,
    grpc_connection_established,
)

import random
# import evidently
# import pandas as pd
# import numpy as np
from collections.abc import Generator
from typing import Optional
import json
import pathlib
import torch
import io

# from evidently import ColumnMapping
# from evidently.report import Report
# from evidently.metrics import EmbeddingsDriftMetric
# from evidently.metrics.data_drift.embedding_drift_methods import model, distance, ratio, mmd
# import PIL

import logging

logger = logging.getLogger(__name__)

# class ModelInfo:
#     def __init__(
#         self,
#         pipeline_id: int,
#         model_name: str,
#         model_config: dict,
#         device: str,
#         amp: bool,
#         base_dir: pathlib.Path
#     ):
#         self.pipeline_id = pipeline_id
#         self.model_name = model_name
#         model_module = dynamic_module_import("modyn.models")
#         self.model_handler = getattr(model_module, self.model_name)
#         self.model_configuration_dict = json.loads(model_config)
#         self.device = device
#         self.amp = amp
#         self._model = self.model_handler(
#             self.model_configuration_dict, self.device, self.amp
#         )

#         self.base_dir = base_dir
#         assert self.base_dir.exists(), f"Temporary Directory {self.base_dir} should have been created."
    
#     def _load_state(self, path: pathlib.Path) -> None:
#         assert path.exists(), "Cannot load state from non-existing file"

#         self._info(f"Loading model state from {path}")
#         with open(path, "rb") as state_file:
#             checkpoint = torch.load(io.BytesIO(state_file.read()), map_location=torch.device("cpu"))

#         assert "model" in checkpoint
#         self._model.model.load_state_dict(checkpoint["model"])

#         # delete trained model from disk
#         path.unlink()
    
#     def download_trained_model(self, model_id: int):
#         fetch_request = FetchModelRequest(model_id=model_id, load_metadata=False)
#         fetch_resp: FetchModelResponse = self._model_storage_stub.FetchModel(fetch_request)

#         if not fetch_resp.success:
#             logger.error(
#                 f"Trained model {model_id} cannot be fetched from model storage. "
#                 f"Evaluation cannot be started."
#             )
#             return EvaluateModelResponse(evaluation_started=False)
#         trained_model_path = download_trained_model(
#             logger=logger,
#             model_storage_config=self._config["model_storage"],
#             remote_path=pathlib.Path(fetch_resp.model_path),
#             checksum=fetch_resp.checksum,
#             identifier=self.pipeline_id,
#             base_directory=self.base_dir,
#         )
#         self._load_state(trained_model_path)


class DataLoaderInfo:
    def __init__(
        self,
        pipeline_id: int,
        dataset_id: str,
        num_dataloaders,
        batch_size,
        bytes_parser,
        transform_list: list[str],
        storage_address,
        selector_address,
        num_prefetched_partitions,
        parallel_prefetch_requests,
        tokenizer
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


class DataDriftTrigger(Trigger):
    """Triggers when a certain number of data points have been used."""

    def __init__(self, trigger_config: dict):
        self.detection_data_points: int = 2
        if "data_points_for_detection" in trigger_config.keys():
            self.detection_data_points = trigger_config["data_points_for_detection"]
        assert self.detection_data_points > 0, "data_points_for_trigger needs to be at least 1"

        self.retrain_threshold: float = 0
        if "retrain_drift_threshold" in trigger_config.keys():
            self.retrain_threshold = trigger_config["retrain_drift_threshold"]
        assert self.retrain_threshold >= 0 and self.retrain_threshold <= 1, "retrain_drift_threshold range [0,1]"

        self.sample_size: int = 100
        if "sample_size" in trigger_config.keys():
            self.sample_size = trigger_config["sample_size"]
        assert self.sample_size > 0, "sample_size needs to be at least 1"

        self.previous_trigger_id: Optional[int] = None
        self.previous_model_id: Optional[int] = None
        # self.model_info: Optional[ModelInfo] = None
        self.dataloader_info: Optional[DataLoaderInfo] = None
        
        self.data_cache = []
        self.cached_data_points = 0
        self.remaining_data_points = 0
        self.trigger_data_points = 0

        super().__init__(trigger_config)
        

    # TODO(JZ)
    def detect_drift(self, idx_start, idx_end) -> float:
        assert self.previous_trigger_id is not None
        assert self.dataloader_info is not None     
        # get training data keys of previous trigger from storage
        logger.info(f"[Prev trigger {self.previous_trigger_id}] Dataloader")
        reference_dataloader = prepare_trigger_dataloader_by_trigger(
            self.dataloader_info.pipeline_id,
            self.previous_trigger_id,
            self.dataloader_info.dataset_id,
            self.dataloader_info.num_dataloaders,
            self.dataloader_info.batch_size,
            self.dataloader_info.bytes_parser,
            self.dataloader_info.transform_list,
            self.dataloader_info.storage_address,
            self.dataloader_info.selector_address,
            self.dataloader_info.training_id,
            self.dataloader_info.num_prefetched_partitions,
            self.dataloader_info.parallel_prefetch_requests,
            tokenizer=self.dataloader_info.tokenizer,
            sample_size=self.sample_size,
        )

        # get new data keys
        # TODO(JZ): remove cache and get data from selector??? check whether trigger exists at selector???
        current_keys, _, _ = zip(*self.data_cache[idx_start : idx_end])  # type: ignore
        logger.info(f"[Next trigger {self.previous_trigger_id + 1}] Dataloader")
        current_dataloader = prepare_trigger_dataloader_given_keys(
            self.dataloader_info.dataset_id,
            self.dataloader_info.num_dataloaders,
            self.dataloader_info.batch_size,
            self.dataloader_info.bytes_parser,
            self.dataloader_info.transform_list,
            self.dataloader_info.storage_address,
            self.previous_trigger_id + 1,
            current_keys,
            self.sample_size,
        )

        # Fetch model used for embedding.
        # Wait until the previous model is available if it's used for embedding
        # Compute embedding: pd.dataframes with colname 'col_i'
        # Run Evidently detection
        return 0


    # the supervisor informs the Trigger about the previous trigger before it calls next()
    def inform(self, new_data: list[tuple[int, int, int]]) -> Generator[int]:
        # add new data to data_cache
        self.data_cache.extend(new_data)
        
        self.cached_data_points = len(self.data_cache)
        total_data_for_detection = len(self.data_cache)
        detection_data_idx_start = 0
        detection_data_idx_end = 0
        while total_data_for_detection >= self.detection_data_points:
            total_data_for_detection -= self.detection_data_points
            self.trigger_data_points += self.detection_data_points
            detection_data_idx_end += self.detection_data_points

            # trigger id doesn't always start from 0, but always increments by 1
            if self.previous_trigger_id is None:
                # if no previous model exists, always trigger
                triggered = True
            else:
                # if exist previous model, detect drift
                # new_data_idx = max(detection_data_idx_start, self.remaining_data_points)
                triggered = (self.detect_drift(detection_data_idx_start, detection_data_idx_end) >= self.retrain_threshold)
            
            if triggered:
                trigger_idx = len(new_data) - (self.cached_data_points - self.trigger_data_points) - 1
                logger.info(f"[Trigger index] {trigger_idx}")
                # update bookkeeping and pointers
                self.cached_data_points -= self.trigger_data_points
                self.trigger_data_points = 0
                detection_data_idx_start = detection_data_idx_end
                yield trigger_idx      

        # remove triggered data
        self.data_cache = self.data_cache[detection_data_idx_start:]
        self.cached_data_points = len(self.data_cache)
        self.remaining_data_points = len(self.data_cache)

    def inform_dataloader_info(self, pipeline_id: int, pipeline_config: dict, modyn_config: dict) -> None:
        if "num_prefetched_partitions" in pipeline_config["training"]:
            num_prefetched_partitions = pipeline_config["training"]["num_prefetched_partitions"]
        else:
            if "prefetched_partitions" in pipeline_config["training"]:
                raise ValueError(
                    "Found `prefetched_partitions` instead of `num_prefetched_partitions`in training configuration."
                    + " Please rename/remove that configuration"
                )
            logger.warning("Number of prefetched partitions not explicitly given in training config - defaulting to 1.")
            num_prefetched_partitions = 1

        if "parallel_prefetch_requests" in pipeline_config["training"]:
            parallel_prefetch_requests = pipeline_config["training"]["parallel_prefetch_requests"]
        else:
            logger.warning(
                "Number of parallel prefetch requests not explicitly given in training config - defaulting to 1."
            )
            parallel_prefetch_requests = 1

        if "tokenizer" in pipeline_config["data"]:
            tokenizer = pipeline_config["data"]["tokenizer"]
        else:
            tokenizer = None

        if "transformations" in pipeline_config["data"]:
            transform_list = pipeline_config["data"]["transformations"]
        else:
            transform_list = []

        self.dataloader_info = DataLoaderInfo(
            pipeline_id,
            dataset_id=pipeline_config["data"]["dataset_id"],
            num_dataloaders=pipeline_config["training"]["dataloader_workers"],
            batch_size=pipeline_config["training"]["batch_size"],
            bytes_parser=pipeline_config["data"]["bytes_parser_function"],
            transform_list=transform_list,
            storage_address=f"{modyn_config['storage']['hostname']}:{modyn_config['storage']['port']}",
            selector_address=f"{modyn_config['selector']['hostname']}:{modyn_config['selector']['port']}",
            num_prefetched_partitions=num_prefetched_partitions,
            parallel_prefetch_requests=parallel_prefetch_requests,
            tokenizer=tokenizer
        )

        self._storage_channel = grpc.insecure_channel(self.dataloader_info.storage_address, options=grpc_common_config())
        if not grpc_connection_established(self._storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at address {self._storage_address}.")
        self._storagestub = StorageStub(self._storage_channel)
        
    def inform_previous_trigger_and_model(self, previous_trigger_id: Optional[int], previous_model_id: Optional[int]) -> None:
        self.previous_trigger_id = previous_trigger_id
        self.previous_model_id = previous_model_id
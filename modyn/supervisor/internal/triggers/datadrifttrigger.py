from modyn.supervisor.internal.triggers.trigger import Trigger
from modyn.supervisor.internal.triggers.trigger_dataset import prepare_trigger_dataloader_given_keys, prepare_trigger_dataloader_by_trigger
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.models.coreset_methods_support import CoresetSupportingModule
from modyn.metadata_database.models import TrainedModel
from modyn.utils import dynamic_module_import
from modyn.common.ftp import download_trained_model

import grpc
# pylint: disable-next=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub

from modyn.utils.utils import (
    BYTES_PARSER_FUNC_NAME,
    grpc_common_config,
    grpc_connection_established,
)

import pandas as pd
from collections.abc import Generator
from typing import Optional, Union
import json
import pathlib
import torch
import io

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import EmbeddingsDriftMetric
from evidently.metrics.data_drift.embedding_drift_methods import model, distance, ratio, mmd
# import PIL

import logging

logger = logging.getLogger(__name__)


class ModelWrapper:
    def __init__(
        self,
        modyn_config: dict,
        pipeline_id: int,
        device: str,
        base_dir: pathlib.Path,
        model_storage_address: str,
    ):
        self.modyn_config = modyn_config
        self.pipeline_id = pipeline_id
        self._device = device
        self._device_type = "cuda" if "cuda" in self._device else "cpu"
        self.base_dir = base_dir
        assert self.base_dir.exists(), f"Temporary Directory {self.base_dir} should have been created."
        self._model_storage_stub = self.connect_to_model_storage(model_storage_address)
        self._amp: Optional[bool] = None
        self._model = None
    
    @staticmethod
    def connect_to_model_storage(model_storage_address: str) -> ModelStorageStub:
        model_storage_channel = grpc.insecure_channel(model_storage_address)
        assert model_storage_channel is not None
        if not grpc_connection_established(model_storage_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to model storage at address {model_storage_address}."
            )
        return ModelStorageStub(model_storage_channel)
    
    def _load_state(self, path: pathlib.Path) -> None:
        assert path.exists(), "Cannot load state from non-existing file"

        # self._info(f"Loading model state from {path}")
        with open(path, "rb") as state_file:
            checkpoint = torch.load(io.BytesIO(state_file.read()), map_location=torch.device("cpu"))

        assert "model" in checkpoint
        self._model.model.load_state_dict(checkpoint["model"])

        # delete trained model from disk
        path.unlink()
    
    def configure(self, model_id: int):
        with MetadataDatabaseConnection(self.modyn_config) as database:
            trained_model: Optional[TrainedModel] = database.session.get(TrainedModel, model_id)
            if not trained_model:
                logger.error(f"Trained model {model_id} does not exist!")
                return
            model_class_name, model_config, amp = database.get_model_configuration(trained_model.pipeline_id)
        self._amp = amp
        model_module = dynamic_module_import("modyn.models")
        self.model_handler = getattr(model_module, model_class_name)
        self.model_configuration_dict = json.loads(model_config)
        self._model = self.model_handler(
            self.model_configuration_dict, self._device, amp
        )
        assert isinstance(self._model.model, CoresetSupportingModule)
    
    def download(self, model_id: int):
        self.configure(model_id)

        fetch_request = FetchModelRequest(model_id=model_id, load_metadata=False)
        fetch_resp: FetchModelResponse = self._model_storage_stub.FetchModel(fetch_request)

        if not fetch_resp.success:
            logger.error(
                f"Trained model {model_id} cannot be fetched from model storage. "
            )
            raise Exception
        trained_model_path = download_trained_model(
            logger=logger,
            model_storage_config=self.modyn_config["model_storage"],
            remote_path=pathlib.Path(fetch_resp.model_path),
            checksum=fetch_resp.checksum,
            identifier=self.pipeline_id,
            base_directory=self.base_dir,
        )
        self._load_state(trained_model_path)
    
    def get_embeddings(self, dataloader) -> list[pd.DataFrame]:
        assert self._model is not None
        all_embeddings = []

        self._model.model.eval()
        self._model.model.embedding_recorder.start_recording()

        with torch.no_grad():
            for batch in dataloader:
                data: Union[torch.Tensor, dict]
                if isinstance(batch[1], torch.Tensor):
                    data = batch[1].to(self._device)
                elif isinstance(batch[1], dict):
                    data: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
                    for name, tensor in batch[1].items():
                        data[name] = tensor.to(self._device)

                # batch_size = target.shape[0]
                with torch.autocast(self._device_type, enabled=self._amp):
                    output = self._model.model(data)
                    embeddings = self._model.model.embedding_recorder.embedding
                    if self._device == "cpu":
                        embeddings_data = embeddings.detach().numpy()
                    else:
                        embeddings_data = embeddings.detach().cpu().numpy()
                    all_embeddings.append(pd.DataFrame(embeddings_data).astype("float"))

        self._model.model.embedding_recorder.end_recording()

        return all_embeddings

    def get_embeddings_evidently_form(self, dataloader) -> pd.DataFrame:
        embeddings = self.get_embeddings(dataloader)
        embeddings_df = pd.concat(embeddings, axis=0)
        embeddings_df.columns = ['col_' + str(x) for x in embeddings_df.columns]
        logger.debug(f"[EMBEDDINGS SHAPE] {embeddings_df.shape}")
        # logger.debug(embeddings_df[:3])
        return embeddings_df


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

        self.retrain_threshold: float = 0.5
        if "retrain_drift_threshold" in trigger_config.keys():
            self.retrain_threshold = trigger_config["retrain_drift_threshold"]
        assert self.retrain_threshold >= 0 and self.retrain_threshold <= 1, "retrain_drift_threshold range [0,1]"

        self.sample_size: int = 100
        if "sample_size" in trigger_config.keys():
            self.sample_size = trigger_config["sample_size"]
        assert self.sample_size > 0, "sample_size needs to be at least 1"

        self.previous_trigger_id: Optional[int] = None
        self.previous_data_points: Optional[int] = None
        self.previous_model_id: Optional[int] = None
        self.model_wrapper: Optional[ModelWrapper] = None
        self.dataloader_info: Optional[DataLoaderInfo] = None
        
        self.data_cache = []
        self.cached_data_points = 0
        self.remaining_data_points = 0
        self.trigger_data_points = 0

        super().__init__(trigger_config)
        

    def detect_drift(self, idx_start, idx_end) -> bool:
        assert self.previous_trigger_id is not None
        assert self.previous_data_points is not None and self.previous_data_points > 0
        assert self.previous_model_id is not None
        assert self.dataloader_info is not None   
  
        # get training data keys of previous trigger from storage
        logger.info(f"[Prev Trigger {self.previous_trigger_id}][Prev Model {self.previous_model_id}] Start drift detection")
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
            data_points_in_trigger=self.previous_data_points,
            sample_size=self.sample_size,
        )

        # get new data
        current_keys, _, _ = zip(*self.data_cache[idx_start : idx_end])  # type: ignore
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

        # Fetch model used for embedding
        # TODO(JZ): custom embedding???
        self.model_wrapper.download(self.previous_model_id)

        # Compute embeddings
        reference_embeddings_df = self.model_wrapper.get_embeddings_evidently_form(reference_dataloader)
        current_embeddings_df = self.model_wrapper.get_embeddings_evidently_form(current_dataloader)
        
        # Run Evidently detection
        column_mapping = ColumnMapping(
            embeddings={'small_subset': reference_embeddings_df.columns}
        )

        report = Report(metrics=[
            EmbeddingsDriftMetric('small_subset',
                                drift_method = model(
                                    threshold = 0.55,
                                    bootstrap = None,
                                    quantile_probability = 0.95,
                                    pca_components = None,
                                    )
                                )
        ])
        report.run(reference_data = reference_embeddings_df, current_data = current_embeddings_df,
                column_mapping = column_mapping)
        result = report.as_dict()
        logger.info(f"[DRIFT] {result}")
        #  >= self.retrain_threshold
        return result["metrics"][0]["result"]["drift_detected"]


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
                triggered = self.detect_drift(detection_data_idx_start, detection_data_idx_end)
            
            if triggered:
                trigger_idx = len(new_data) - (self.cached_data_points - self.trigger_data_points) - 1
                logger.info(f">>>>>>>>>>>>>>>>> [Trigger index] {trigger_idx}")
                logger.info(f">>>>>>>>>>>>>>>>> [Trigger data points] {self.trigger_data_points}")
                # update bookkeeping and pointers
                self.cached_data_points -= self.trigger_data_points
                self.trigger_data_points = 0
                detection_data_idx_start = detection_data_idx_end
                yield trigger_idx      

        # remove triggered data
        self.data_cache = self.data_cache[detection_data_idx_start:]
        self.cached_data_points = len(self.data_cache)
        self.remaining_data_points = len(self.data_cache)

    def init_dataloader_info(self, pipeline_id: int, pipeline_config: dict, modyn_config: dict) -> None:
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
        
    def init_model_wrapper(self, pipeline_id: int, pipeline_config: dict, modyn_config: dict, base_dir: pathlib.Path):
        self.model_wrapper = ModelWrapper(
            modyn_config,
            pipeline_id, 
            pipeline_config["training"]["device"], 
            base_dir,
            f"{modyn_config['model_storage']['hostname']}:{modyn_config['model_storage']['port']}"
        )
        
    def inform_previous_trigger_and_model(self, previous_trigger_id: Optional[int], previous_model_id: Optional[int]) -> None:
        self.previous_trigger_id = previous_trigger_id
        self.previous_model_id = previous_model_id
    
    def inform_previous_trigger_data_points(self, previous_trigger_id: int, data_points: int) -> None:
        assert self.previous_trigger_id == previous_trigger_id
        self.previous_data_points = data_points
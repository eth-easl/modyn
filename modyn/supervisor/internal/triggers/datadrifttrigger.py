import json
import logging
import os
import pathlib
from typing import Generator, Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from modyn.supervisor.internal.triggers.model_downloader import ModelDownloader

# pylint: disable-next=no-name-in-module
from modyn.supervisor.internal.triggers.trigger import Trigger
from modyn.supervisor.internal.triggers.trigger_datasets import DataLoaderInfo
from modyn.supervisor.internal.triggers.utils import (
    get_embeddings_evidently_format,
    get_evidently_metrics,
    prepare_trigger_dataloader_by_trigger,
    prepare_trigger_dataloader_fixed_keys,
)

logger = logging.getLogger(__name__)


class DataDriftTrigger(Trigger):
    """Triggers when a certain number of data points have been used."""

    def __init__(self, trigger_config: dict):
        self.pipeline_id: Optional[int] = None
        self.pipeline_config: Optional[dict] = None
        self.modyn_config: Optional[dict] = None
        self.base_dir: Optional[pathlib.Path] = None

        self.previous_trigger_id: Optional[int] = None
        self.previous_model_id: Optional[int] = None
        self.previous_data_points: Optional[int] = None
        self.model_updated: bool = False

        self.dataloader_info: Optional[DataLoaderInfo] = None
        self.model_downloader: Optional[ModelDownloader] = None

        self.detection_interval: int = 1000
        self.sample_size: Optional[int] = None
        self.evidently_column_mapping_name = "data"
        self.metrics: Optional[list] = None

        self.exp_output_dir: Optional[pathlib.Path] = None
        self.drift_dir: Optional[pathlib.Path] = None

        self.data_cache: list[tuple[int, int, int]] = []
        self.leftover_data_points = 0

        if len(trigger_config) > 0:
            self._parse_trigger_config(trigger_config)

        super().__init__(trigger_config)

    def _parse_trigger_config(self, trigger_config: dict) -> None:
        if "data_points_for_detection" in trigger_config.keys():
            self.detection_interval = trigger_config["data_points_for_detection"]
        assert self.detection_interval > 0, "data_points_for_detection needs to be at least 1"

        if "sample_size" in trigger_config.keys():
            self.sample_size = trigger_config["sample_size"]
        assert self.sample_size is None or self.sample_size > 0, "sample_size needs to be at least 1"

        self.metrics = get_evidently_metrics(self.evidently_column_mapping_name, trigger_config)

    def _init_dataloader_info(self) -> None:
        assert self.pipeline_config is not None
        assert self.modyn_config is not None

        training_config = self.pipeline_config["training"]
        data_config = self.pipeline_config["data"]

        if "num_prefetched_partitions" in training_config:
            num_prefetched_partitions = training_config["num_prefetched_partitions"]
        else:
            if "prefetched_partitions" in training_config:
                raise ValueError(
                    "Found `prefetched_partitions` instead of `num_prefetched_partitions`in training configuration."
                    + " Please rename/remove that configuration"
                )
            logger.warning("Number of prefetched partitions not explicitly given in training config - defaulting to 1.")
            num_prefetched_partitions = 1

        if "parallel_prefetch_requests" in training_config:
            parallel_prefetch_requests = training_config["parallel_prefetch_requests"]
        else:
            logger.warning(
                "Number of parallel prefetch requests not explicitly given in training config - defaulting to 1."
            )
            parallel_prefetch_requests = 1

        if "tokenizer" in data_config:
            tokenizer = data_config["tokenizer"]
        else:
            tokenizer = None

        if "transformations" in data_config:
            transform_list = data_config["transformations"]
        else:
            transform_list = []

        self.dataloader_info = DataLoaderInfo(
            self.pipeline_id,
            dataset_id=data_config["dataset_id"],
            num_dataloaders=training_config["dataloader_workers"],
            batch_size=training_config["batch_size"],
            bytes_parser=data_config["bytes_parser_function"],
            transform_list=transform_list,
            storage_address=f"{self.modyn_config['storage']['hostname']}:{self.modyn_config['storage']['port']}",
            selector_address=f"{self.modyn_config['selector']['hostname']}:{self.modyn_config['selector']['port']}",
            num_prefetched_partitions=num_prefetched_partitions,
            parallel_prefetch_requests=parallel_prefetch_requests,
            tokenizer=tokenizer,
        )

    def _init_model_downloader(self) -> None:
        assert self.pipeline_config is not None
        assert self.modyn_config is not None

        self.model_downloader = ModelDownloader(
            self.modyn_config,
            self.pipeline_id,
            self.pipeline_config["training"]["device"],
            self.base_dir,
            f"{self.modyn_config['model_storage']['hostname']}:{self.modyn_config['model_storage']['port']}",
        )

    def _create_dirs(self) -> None:
        assert self.pipeline_id is not None
        assert self.base_dir is not None

        self.exp_output_dir = self.base_dir / str(self.pipeline_id) / "datadrift"
        self.drift_dir = self.exp_output_dir / "drift_results"
        os.makedirs(self.drift_dir, exist_ok=True)

    def init_trigger(self, pipeline_id: int, pipeline_config: dict, modyn_config: dict, base_dir: pathlib.Path) -> None:
        self.pipeline_id = pipeline_id
        self.pipeline_config = pipeline_config
        self.modyn_config = modyn_config
        self.base_dir = base_dir

        self._init_dataloader_info()
        self._init_model_downloader()
        self._create_dirs()

    def run_detection(self, reference_embeddings_df: pd.DataFrame, current_embeddings_df: pd.DataFrame) -> bool:
        assert self.dataloader_info is not None
        assert self.drift_dir is not None

        # Run Evidently detection
        column_mapping = ColumnMapping(embeddings={self.evidently_column_mapping_name: reference_embeddings_df.columns})

        # https://docs.evidentlyai.com/user-guide/customization/embeddings-drift-parameters
        report = Report(metrics=self.metrics)
        report.run(
            reference_data=reference_embeddings_df, current_data=current_embeddings_df, column_mapping=column_mapping
        )
        result = report.as_dict()
        result_print = [
            (x["result"]["drift_score"], x["result"]["method_name"], x["result"]["drift_detected"])
            for x in result["metrics"]
        ]
        logger.info(
            f"[DataDriftDetector][Prev Trigger {self.previous_trigger_id}][Dataset {self.dataloader_info.dataset_id}]"
            + f"[Result] {result_print}"
        )

        with open(self.drift_dir / f"drift_{self.previous_trigger_id}.json", "w", encoding="utf-8") as f:
            json.dump(result, f)

        return result["metrics"][0]["result"]["drift_detected"]

    def detect_drift(self, idx_start: int, idx_end: int) -> bool:
        assert self.previous_trigger_id is not None
        assert self.previous_data_points is not None and self.previous_data_points > 0
        assert self.previous_model_id is not None
        assert self.dataloader_info is not None
        assert self.model_downloader is not None

        reference_dataloader = prepare_trigger_dataloader_by_trigger(
            self.previous_trigger_id,
            self.dataloader_info,
            data_points_in_trigger=self.previous_data_points,
            sample_size=self.sample_size,
        )

        current_keys, _, _ = zip(*self.data_cache[idx_start:idx_end])  # type: ignore
        current_dataloader = prepare_trigger_dataloader_fixed_keys(
            self.previous_trigger_id + 1,
            self.dataloader_info,
            current_keys,
            sample_size=self.sample_size,
        )

        # Download previous model as embedding encoder
        # TODO(366) Support custom model as embedding encoder
        if self.model_updated:
            self.model_downloader.download(self.previous_model_id)
            self.model_updated = False

        # Compute embeddings
        reference_embeddings_df = get_embeddings_evidently_format(self.model_downloader, reference_dataloader)
        current_embeddings_df = get_embeddings_evidently_format(self.model_downloader, current_dataloader)

        drift_detected = self.run_detection(reference_embeddings_df, current_embeddings_df)

        return drift_detected

    def inform(self, new_data: list[tuple[int, int, int]]) -> Generator[int, None, None]:
        """
        Use Generator here because this data drift trigger
        needs to wait for the previous trigger to finish and get the model
        """
        # add new data to data_cache
        self.data_cache.extend(new_data)

        unvisited_data_points = len(self.data_cache)
        untriggered_data_points = unvisited_data_points
        # the sliding window of data points for detection
        detection_idx_start = 0
        detection_idx_end = 0

        while unvisited_data_points >= self.detection_interval:
            unvisited_data_points -= self.detection_interval
            detection_idx_end += self.detection_interval
            if detection_idx_end <= self.leftover_data_points:
                continue

            # trigger_id doesn't always start from 0
            if self.previous_trigger_id is None:
                # if no previous trigger exists, always trigger retraining
                triggered = True
            else:
                # if exist previous trigger, detect drift
                triggered = self.detect_drift(detection_idx_start, detection_idx_end)

            if triggered:
                trigger_data_points = detection_idx_end - detection_idx_start
                trigger_idx = len(new_data) - (untriggered_data_points - trigger_data_points) - 1

                # update bookkeeping and sliding window
                untriggered_data_points -= trigger_data_points
                detection_idx_start = detection_idx_end
                yield trigger_idx

        # remove triggered data from cache
        del self.data_cache[:detection_idx_start]
        self.leftover_data_points = detection_idx_end - detection_idx_start

    def inform_previous_trigger_and_data_points(self, previous_trigger_id: int, data_points: int) -> None:
        self.previous_trigger_id = previous_trigger_id
        self.previous_data_points = data_points

    def inform_previous_model(self, previous_model_id: int) -> None:
        self.previous_model_id = previous_model_id
        self.model_updated = True

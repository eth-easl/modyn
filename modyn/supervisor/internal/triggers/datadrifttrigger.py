from __future__ import annotations

import logging
from typing import Generator, Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import EmbeddingsDriftMetric
from evidently.metrics.data_drift import embedding_drift_methods
from evidently.report import Report
from modyn.config.schema.pipeline import DataDriftTriggerConfig
from modyn.supervisor.internal.triggers.embedding_encoder_utils import EmbeddingEncoder, EmbeddingEncoderDownloader

# pylint: disable-next=no-name-in-module
from modyn.supervisor.internal.triggers.trigger import Trigger, TriggerContext
from modyn.supervisor.internal.triggers.trigger_datasets import DataLoaderInfo
from modyn.supervisor.internal.triggers.utils import (
    convert_tensor_to_df,
    get_embeddings,
    prepare_trigger_dataloader_by_trigger,
    prepare_trigger_dataloader_fixed_keys,
)

logger = logging.getLogger(__name__)


class DataDriftTrigger(Trigger):
    """Triggers when a certain number of data points have been used."""

    def __init__(self, config: DataDriftTriggerConfig):
        self.config = config
        self.context: TriggerContext | None = None

        self.previous_trigger_id: Optional[int] = None
        self.previous_model_id: Optional[int] = None
        self.previous_data_points: Optional[int] = None
        self.model_updated: bool = False

        self.dataloader_info: Optional[DataLoaderInfo] = None
        self.encoder_downloader: Optional[EmbeddingEncoderDownloader] = None
        self.embedding_encoder: Optional[EmbeddingEncoder] = None

        self.evidently_column_mapping_name = "data"
        self.metrics = self._get_evidently_metrics(self.evidently_column_mapping_name, config)

        self.data_cache: list[tuple[int, int, int]] = []
        self.leftover_data_points = 0

    def init_trigger(self, context: TriggerContext) -> None:
        self.context = context
        self._init_dataloader_info()
        self._init_encoder_downloader()

    def run_detection(self, reference_embeddings_df: pd.DataFrame, current_embeddings_df: pd.DataFrame) -> bool:
        assert self.dataloader_info is not None

        # Run Evidently detection
        # ColumnMapping is {mapping name: column indices},
        # an Evidently way of identifying (sub)columns to use in the detection.
        # e.g. {"even columns": [0,2,4]}.
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

        return result["metrics"][0]["result"]["drift_detected"]

    def detect_drift(self, idx_start: int, idx_end: int) -> bool:
        """Compare current data against reference data.
        current data: all untriggered samples in the sliding window in inform().
        reference data: the training samples of the previous trigger.
        Get the dataloaders, download the embedding encoder model if necessary,
        compute embeddings of current and reference data, then run detection on the embeddings.
        """
        assert self.previous_trigger_id is not None
        assert self.previous_data_points is not None and self.previous_data_points > 0
        assert self.previous_model_id is not None
        assert self.dataloader_info is not None
        assert self.encoder_downloader is not None
        assert self.context and self.context.pipeline_config is not None

        reference_dataloader = prepare_trigger_dataloader_by_trigger(
            self.previous_trigger_id,
            self.dataloader_info,
            data_points_in_trigger=self.previous_data_points,
            sample_size=self.config.sample_size,
        )

        current_keys, _, _ = zip(*self.data_cache[idx_start:idx_end])  # type: ignore
        current_dataloader = prepare_trigger_dataloader_fixed_keys(
            self.previous_trigger_id + 1,
            self.dataloader_info,
            current_keys,  # type: ignore
            sample_size=self.config.sample_size,
        )

        # Download previous model as embedding encoder
        # TODO(417) Support custom model as embedding encoder
        if self.model_updated:
            self.embedding_encoder = self.encoder_downloader.setup_encoder(
                self.previous_model_id, self.context.pipeline_config.training.device
            )
            self.model_updated = False

        # Compute embeddings
        assert self.embedding_encoder is not None
        reference_embeddings = get_embeddings(self.embedding_encoder, reference_dataloader)
        current_embeddings = get_embeddings(self.embedding_encoder, current_dataloader)
        reference_embeddings_df = convert_tensor_to_df(reference_embeddings, "col_")
        current_embeddings_df = convert_tensor_to_df(current_embeddings, "col_")

        drift_detected = self.run_detection(reference_embeddings_df, current_embeddings_df)

        return drift_detected

    def inform(self, new_data: list[tuple[int, int, int]]) -> Generator[int, None, None]:
        """The DataDriftTrigger takes a batch of new data as input. It adds the new data to its data_cache.
        Then, it iterates through the data_cache with a sliding window of current data for drift detection.
        The sliding window is determined by two pointers: detection_idx_start and detection_idx_end.
        We fix the start pointer and advance the end pointer by detection_interval in every iteration.
        In every iteration we run data drift detection on the sliding window of current data.
        Note, if we have remaining untriggered data from the previous batch of new data,
        we include all of them in the first drift detection.
        The remaining untriggered data has been processed in the previous new data batch,
        so there's no need to run detection only on remaining data in this batch.
        If a retraining is triggered, all data in the sliding window becomes triggering data. Advance the start ptr.
        After traversing the data_cache, we remove all the triggered data from the cache
        and record the number of remaining untriggered samples.
        Use Generator here because this data drift trigger
        needs to wait for the previous trigger to finish and get the model.
        """
        # add new data to data_cache
        self.data_cache.extend(new_data)

        unvisited_data_points = len(self.data_cache)
        untriggered_data_points = unvisited_data_points
        # the sliding window of data points for detection
        detection_idx_start = 0
        detection_idx_end = 0

        while unvisited_data_points >= self.config.detection_interval_data_points:
            unvisited_data_points -= self.config.detection_interval_data_points
            detection_idx_end += self.config.detection_interval_data_points
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
                # Index of the last sample of the trigger. Index is relative to the new_data list.
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

    # --------------------------------------------------- INTERNAL --------------------------------------------------- #

    def _get_evidently_metrics(
        self, column_mapping_name: str, config: DataDriftTriggerConfig
    ) -> list[EmbeddingsDriftMetric]:
        """This function instantiates an Evidently metric given metric configuration.
        If we want to support multiple metrics in the future, we can loop through the configurations.

        Evidently metric configurations follow exactly the four DriftMethods defined in embedding_drift_methods:
        model, distance, mmd, ratio
        If metric_name not given, we use the default 'model' metric.
        Otherwise, we use the metric given by metric_name, with optional metric configuration specific to the metric.
        """
        metric = getattr(embedding_drift_methods, config.metric)(**config.metric_config)
        metrics = [EmbeddingsDriftMetric(column_mapping_name, drift_method=metric)]
        return metrics

    def _init_dataloader_info(self) -> None:
        assert self.context

        training_config = self.context.pipeline_config.training
        data_config = self.context.pipeline_config.data

        self.dataloader_info = DataLoaderInfo(
            self.context.pipeline_id,
            dataset_id=data_config.dataset_id,
            num_dataloaders=training_config.dataloader_workers,
            batch_size=training_config.batch_size,
            bytes_parser=data_config.bytes_parser_function,
            transform_list=data_config.transformations,
            storage_address=f"{self.context.modyn_config.storage.address}",
            selector_address=f"{self.context.modyn_config.selector.address}",
            num_prefetched_partitions=training_config.num_prefetched_partitions,
            parallel_prefetch_requests=training_config.parallel_prefetch_requests,
            shuffle=training_config.shuffle,
            tokenizer=data_config.tokenizer,
        )

    def _init_encoder_downloader(self) -> None:
        assert self.context is not None

        self.encoder_downloader = EmbeddingEncoderDownloader(
            self.context.modyn_config,
            self.context.pipeline_id,
            self.context.base_dir,
            f"{self.context.modyn_config.modyn_model_storage.address}",
        )

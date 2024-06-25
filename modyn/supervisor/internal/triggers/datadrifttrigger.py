from __future__ import annotations

import logging
from typing import Generator, Optional

from modyn.config.schema.pipeline import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.drift.result import MetricResult
from modyn.supervisor.internal.triggers.drift.alibi_detector import AlibiDriftDetector
from modyn.supervisor.internal.triggers.drift.evidently_detector import EvidentlyDriftDetector
from modyn.supervisor.internal.triggers.embedding_encoder_utils import EmbeddingEncoder, EmbeddingEncoderDownloader

# pylint: disable-next=no-name-in-module
from modyn.supervisor.internal.triggers.models import DriftTriggerEvalLog, TriggerPolicyEvaluationLog
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

        self.data_cache: list[tuple[int, int, int]] = []
        self.leftover_data_points = 0

        self.evidently_detector = EvidentlyDriftDetector(config.metrics)
        self.alibi_detector = AlibiDriftDetector(config.metrics)

    def init_trigger(self, context: TriggerContext) -> None:
        self.context = context
        self._init_dataloader_info()
        self._init_encoder_downloader()

    def inform(
        self, new_data: list[tuple[int, int, int]], log: TriggerPolicyEvaluationLog | None = None
    ) -> Generator[int, None, None]:
        """Decides whether to trigger retraining on a batch of new data based on data drift.

        New data is stored in data_cache and iterated over via a sliding window of current data for drift detection.

        The sliding window is determined by two pointers: detection_idx_start and detection_idx_end.
        We hold the start pointer constant and advance the end pointer by detection_interval in every iteration.
        In every iteration we run data drift detection on the sliding window of current data.

        Note: When having remaining untriggered data from the previous batch of new data,
        we include all of them in the first drift detection.

        The remaining untriggered data has been processed in the previous new data batch,
        so there's no need to run detection only on remaining data in this batch.
        If a retraining is triggered, all data in the sliding window becomes triggering data.
        Advance the start ptr after traversing the data_cache, we remove all the triggered data from the cache
        and record the number of remaining untriggered samples.

        Args:
            new_data: List of new data (keys, timestamps, labels)
            log: The log to store the trigger policy evaluation results to be able to verify trigger decisions

        Returns:
            a generator here that waits for the previous trigger to finish and get the model.
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

            if self.previous_trigger_id is None:
                triggered = True  # if no previous trigger exists, always trigger retraining
                drift_results: dict[str, MetricResult] = {}
            else:
                # if exist previous trigger, detect drift
                triggered, drift_results = self._run_detection(detection_idx_start, detection_idx_end)

            drift_eval_log = DriftTriggerEvalLog(
                detection_idx_start=detection_idx_start,
                detection_idx_end=detection_idx_end,
                triggered=triggered,
                trigger_index=-1,
                drift_results=drift_results,
            )

            if triggered:
                trigger_data_points = detection_idx_end - detection_idx_start
                # Index of the last sample of the trigger. Index is relative to the new_data list.
                trigger_idx = len(new_data) - (untriggered_data_points - trigger_data_points) - 1

                # log
                drift_eval_log.data_points = trigger_data_points
                if log:
                    log.evaluations.append(drift_eval_log)

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

    def _run_detection(self, idx_start: int, idx_end: int) -> tuple[bool, dict[str, MetricResult]]:
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

        drift_results = {
            **self.evidently_detector.detect_drift(reference_embeddings_df, current_embeddings_df),
            **self.alibi_detector.detect_drift(reference_embeddings, current_embeddings),
        }
        logger.info(
            f"[DataDriftDetector][Prev Trigger {self.previous_trigger_id}][Dataset {self.dataloader_info.dataset_id}]"
            + f"[Result] {drift_results}"
        )

        # aggregate the different drift detection results
        drift_detected = self.config.aggregation_strategy.aggregate_decision_func(drift_results)

        return drift_detected, drift_results

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

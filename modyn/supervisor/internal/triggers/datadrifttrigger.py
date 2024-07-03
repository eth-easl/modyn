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
    prepare_trigger_dataloader_fixed_keys,
)

logger = logging.getLogger(__name__)


class DataDriftTrigger(Trigger):
    """Triggers when a certain number of data points have been used."""

    def __init__(self, config: DataDriftTriggerConfig):
        self.config = config
        self.context: TriggerContext | None = None

        self.previous_model_id: Optional[int] = None
        self.model_updated: bool = False

        self.dataloader_info: Optional[DataLoaderInfo] = None
        self.encoder_downloader: Optional[EmbeddingEncoderDownloader] = None
        self.embedding_encoder: Optional[EmbeddingEncoder] = None

        self.evidently_detector = EvidentlyDriftDetector(config.metrics)
        self.alibi_detector = AlibiDriftDetector(config.metrics)

        self._reference_window: list[tuple[int, int]] = []
        self._current_window: list[tuple[int, int]] = []
        self._total_items_in_current_window = 0
        self._triggered_once = False

    def init_trigger(self, context: TriggerContext) -> None:
        self.context = context
        self._init_dataloader_info()
        self._init_encoder_downloader()

    def _update_curr_window(self, new_data: list[tuple[int, int]]) -> None:
        self._current_window.extend(new_data)
        self._total_items_in_current_window += len(new_data)

        if self.config.windowing_strategy.id == "AmountWindowingStrategy":
            if len(self._current_window) > self.config.windowing_strategy.amount:
                self._current_window = self._current_window[self.config.windowing_strategy.amount :]
        elif self.config.windowing_strategy.id == "TimeWindowingStrategy":
            highest_timestamp = new_data[-1][1]
            cutoff = highest_timestamp - self.config.windowing_strategy.limit_seconds
            self._current_window = [(key, timestamp) for key, timestamp in self._current_window if timestamp >= cutoff]
        else:
            raise NotImplementedError(f"{self.config.windowing_strategy.id} is not implemented!")

    def _handle_drift_result(
        self,
        triggered: bool,
        trigger_idx: int,
        drift_results: dict[str, MetricResult],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> Generator[int, None, None]:
        drift_eval_log = DriftTriggerEvalLog(
            detection_idx_start=self._current_window[0],
            detection_idx_end=self._current_window[-1],
            triggered=triggered,
            trigger_index=-1,
            drift_results=drift_results,
        )

        if triggered:
            self._reference_window = self._current_window  # Current assumption: same windowing strategy on both
            self._current_window = [] if self.config.reset_current_window_on_trigger else self._current_window
            self._total_items_in_current_window = (
                0 if self.config.reset_current_window_on_trigger else self._total_items_in_current_window
            )

            if log:
                log.evaluations.append(drift_eval_log)

            yield trigger_idx

    def inform(
        self, new_data: list[tuple[int, int, int]], log: TriggerPolicyEvaluationLog | None = None
    ) -> Generator[int, None, None]:
        """Decides whether to trigger retraining on a batch of new data based on data drift.

            We keep a reference and a current window. TODO: write ddocstring
        Args:
            new_data: List of new data (keys, timestamps, labels)
            log: The log to store the trigger policy evaluation results to be able to verify trigger decisions
        """
        new_key_ts = [(key, timestamp) for key, timestamp, _ in new_data]
        detect_interval = self.config.detection_interval_data_points
        offset = self._total_items_in_current_window % detect_interval

        if offset + len(new_key_ts) < detect_interval:
            # No detection in this trigger
            self._update_curr_window(new_key_ts)
            return

        # At least one detection, fill up window up to that detection
        self._update_curr_window(new_key_ts[: detect_interval - offset])
        new_key_ts = new_key_ts[detect_interval - offset :]
        trigger_idx = detect_interval - offset - 1  # If we trigger, it will be on this index

        if not self._triggered_once:
            # If we've never triggered before, always trigger
            self._triggered_once = True
            triggered = True
            drift_results: dict[str, MetricResult] = {}
        else:
            # Run the detection
            triggered, drift_results = self._run_detection()

        yield from self._handle_drift_result(triggered, trigger_idx, drift_results, log=log)

        # Go through remaining data in new data in batches of `detect_interval`
        for i in range(0, len(new_key_ts), detect_interval):
            batch = new_key_ts[i : i + detect_interval]
            trigger_idx += detect_interval
            self._update_curr_window(batch)

            if len(batch) == detect_interval:
                # Regular batch, in this case run detection
                triggered, drift_results = self._run_detection()
                yield from self._handle_drift_result(triggered, trigger_idx, drift_results, log=log)

    def inform_previous_model(self, previous_model_id: int) -> None:
        self.previous_model_id = previous_model_id
        self.model_updated = True

    # --------------------------------------------------- INTERNAL --------------------------------------------------- #

    def _run_detection(self) -> tuple[bool, dict[str, MetricResult]]:
        """Compare current data against reference data.
        current data: all untriggered samples in the sliding window in inform().
        reference data: the training samples of the previous trigger.
        Get the dataloaders, download the embedding encoder model if necessary,
        compute embeddings of current and reference data, then run detection on the embeddings.
        """
        assert self.previous_model_id is not None
        assert self.dataloader_info is not None
        assert self.encoder_downloader is not None
        assert self.context and self.context.pipeline_config is not None
        assert len(self._reference_window) > 0
        assert len(self._current_window) > 0

        reference_dataloader = prepare_trigger_dataloader_fixed_keys(
            self.dataloader_info, [key for key, _ in self._reference_window]
        )

        current_dataloader = prepare_trigger_dataloader_fixed_keys(
            self.dataloader_info, [key for key, _ in self._current_window]
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
        logger.info(f"[DataDriftDetector][Dataset {self.dataloader_info.dataset_id}]" + f"[Result] {drift_results}")

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

from __future__ import annotations

import gc
import logging
from collections.abc import Generator

from numpy import mean

from modyn.config.schema.pipeline import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.drift.detection_window import (
    AmountWindowingStrategy,
    DriftWindowingStrategy,
    TimeWindowingStrategy,
)
from modyn.config.schema.pipeline.trigger.drift.metric import ThresholdDecisionCriterion
from modyn.config.schema.pipeline.trigger.drift.result import MetricResult
from modyn.supervisor.internal.triggers.drift.decision_policy import (
    DriftDecisionPolicy,
    DynamicDecisionPolicy,
    ThresholdDecisionPolicy,
)
from modyn.supervisor.internal.triggers.drift.detection_window.amount import (
    AmountDetectionWindows,
)
from modyn.supervisor.internal.triggers.drift.detection_window.time_ import (
    TimeDetectionWindows,
)
from modyn.supervisor.internal.triggers.drift.detection_window.window import (
    DetectionWindows,
)
from modyn.supervisor.internal.triggers.drift.detector.alibi import AlibiDriftDetector
from modyn.supervisor.internal.triggers.drift.detector.evidently import (
    EvidentlyDriftDetector,
)
from modyn.supervisor.internal.triggers.embedding_encoder_utils import (
    EmbeddingEncoder,
    EmbeddingEncoderDownloader,
)

# pylint: disable-next=no-name-in-module
from modyn.supervisor.internal.triggers.models import (
    DriftTriggerEvalLog,
    TriggerPolicyEvaluationLog,
)
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

        self.previous_model_id: int | None = None
        self.model_updated: bool = False

        self.dataloader_info: DataLoaderInfo | None = None
        self.encoder_downloader: EmbeddingEncoderDownloader | None = None
        self.embedding_encoder: EmbeddingEncoder | None = None

        self._sample_left_until_detection = (
            config.detection_interval_data_points
        )  # allows to detect drift in a fixed interval
        self._windows = _setup_detection_window_manager(config.windowing_strategy)
        self._triggered_once = False

        self.evidently_detector = EvidentlyDriftDetector(config.metrics)
        self.alibi_detector = AlibiDriftDetector(config.metrics)

        # Every decision policy wraps one metric and is responsible for making decisions based on the metric's results
        # and the metric's range of distance values
        self.decision_policies = _setup_decision_engines(config)

        # list of reference windows for each warmup interval
        self.warmup_intervals: list[list[tuple[int, int]]] = []

    def init_trigger(self, context: TriggerContext) -> None:
        self.context = context
        self._init_dataloader_info()
        self._init_encoder_downloader()

    def _handle_drift_result(
        self,
        triggered: bool,
        trigger_idx: int,
        drift_results: dict[str, MetricResult],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> Generator[int, None, None]:
        drift_eval_log = DriftTriggerEvalLog(
            detection_interval=(
                self._windows.current[0][1],
                self._windows.current[-1][1],
            ),
            reference_interval=(
                (self._windows.reference[0][1], self._windows.reference[-1][1]) if self._windows.reference else (-1, -1)
            ),
            triggered=triggered,
            trigger_index=-1,
            drift_results=drift_results,
        )

        if triggered:
            self._windows.inform_trigger()

            if log:
                log.evaluations.append(drift_eval_log)

            yield trigger_idx

    def inform(
        self,
        new_data: list[tuple[int, int, int]],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> Generator[int, None, None]:
        """Analyzes a batch of new data to determine if data drift has occurred
        and triggers retraining if necessary.

        This method maintains a reference window and a current window of data points. The reference window contains
        data points from the period before the last detected drift, while the current window accumulates incoming data
        points until a detection interval is reached. When the number of data points in the current window reaches the
        detection interval threshold, drift detection is performed. If drift is detected, the reference window is
        updated with the current window's data, and depending on the configuration, the current window is either reset
        or extended for the next round of detection.

        The method works as follows:
        1. Extract keys and timestamps from the incoming data points.
        2. Use the offset, which is the number of data points in the current window that have not yet contributed
        to a drift detection.
        3. If the sum of the offset and the length of the new data is less than the detection interval, update the
        current window with the new data and return without performing drift detection.
        4. If the detection interval is reached, update the current window up to the point of detection and perform
        drift detection.
        5. If drift is detected or if it's the first run (and thus always triggers), handle the drift result by
        updating the reference window and possibly resetting or extending the current window.
        6. Continue processing any remaining new data in batches according to the detection interval, performing
        drift detection on each batch.

        Note: The method is a generator and must be iterated over to execute its logic.
        Args:
            new_data: A list of new data points,
                where each data point is a tuple containing a key, a timestamp, and a label.
            log: An optional log object to store the results of the trigger policy evaluation.

        Yields:
            The index of the data point that triggered the drift detection. This is used to identify the point in the
            data stream where the model's performance may have started to degrade due to drift.
        """
        # pylint: disable=too-many-nested-blocks

        new_key_ts = [(key, timestamp) for key, timestamp, _ in new_data]

        # index of the first unprocessed data point in the batch
        processing_head_in_batch = 0

        # Go through remaining data in new data in batches of `detect_interval`
        while True:
            if self._sample_left_until_detection - len(new_key_ts) > 0:
                # No detection in this trigger because of too few data points to fill detection interval
                self._windows.inform_data(new_key_ts)  # update current window
                self._sample_left_until_detection -= len(new_key_ts)
                return

            # At least one detection, fill up window up to that detection
            next_detection_interval = new_key_ts[: self._sample_left_until_detection]
            self._windows.inform_data(next_detection_interval)

            # Update the remaining data
            processing_head_in_batch += len(next_detection_interval)
            new_key_ts = new_key_ts[len(next_detection_interval) :]

            # Reset for next detection
            self._sample_left_until_detection = self.config.detection_interval_data_points

            # The first ever detection will always trigger
            if not self._triggered_once or len(self.warmup_intervals) < (self.config.warmup_intervals or 0):
                # If we've never triggered before, always trigger
                self._triggered_once = True
                triggered = True
                drift_results: dict[str, MetricResult] = {}
                if len(self.warmup_intervals) < (self.config.warmup_intervals or 0):
                    self.warmup_intervals.append(list(self._windows.reference))

            else:
                # Run the detection

                # if this is the first non warmup detection, we inform the metrics that use decision criteria
                # with calibration requirements about the warmup intervals so they can calibrate their thresholds
                if len(self.warmup_intervals) > 0:
                    # we can ignore the results as the decision criteria will keep track of the warmup results
                    # internally
                    if self._any_metric_needs_calibration():
                        for warmup_interval in self.warmup_intervals:
                            # we generate the calibration with different reference windows, the latest model and
                            # the current window
                            _warmup_triggered, _warmup_results = self._run_detection(
                                warmup_interval,
                                list(self._windows.current),
                                is_warmup=True,
                            )
                            if log:
                                warmup_log = DriftTriggerEvalLog(
                                    detection_interval=(
                                        self._windows.current[0][1],
                                        self._windows.current[-1][1],
                                    ),
                                    reference_interval=(
                                        self._windows.reference[0][1],
                                        self._windows.reference[-1][1],
                                    ),
                                    triggered=_warmup_triggered,
                                    trigger_index=-1,
                                    drift_results=_warmup_results,
                                )
                                log.evaluations.append(warmup_log)

                    self.warmup_intervals = []  # free the memory
                    gc.collect()

                triggered, drift_results = self._run_detection(
                    list(self._windows.reference),
                    list(self._windows.current),
                    is_warmup=False,
                )

            if triggered:
                trigger_idx = processing_head_in_batch - 1
                yield from self._handle_drift_result(triggered, trigger_idx, drift_results, log=log)

    def inform_previous_model(self, previous_model_id: int) -> None:
        self.previous_model_id = previous_model_id
        self.model_updated = True

    # --------------------------------------------------- INTERNAL --------------------------------------------------- #

    def _run_detection(
        self,
        reference: list[tuple[int, int]],
        current: list[tuple[int, int]],
        is_warmup: bool,
    ) -> tuple[bool, dict[str, MetricResult]]:
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
        assert len(reference) > 0
        assert len(current) > 0

        reference_dataloader = prepare_trigger_dataloader_fixed_keys(
            self.dataloader_info, [key for key, _ in reference]
        )

        current_dataloader = prepare_trigger_dataloader_fixed_keys(self.dataloader_info, [key for key, _ in current])

        # Download previous model as embedding encoder
        # TODO(417) Support custom model as embedding encoder
        if self.model_updated:
            self.embedding_encoder = self.encoder_downloader.setup_encoder(
                self.previous_model_id, self.context.pipeline_config.training.device
            )
            self.model_updated = False

        # Compute embeddings
        assert self.embedding_encoder is not None

        # tbd. reuse the embeddings as long as the reference window is not updated
        reference_embeddings = get_embeddings(self.embedding_encoder, reference_dataloader)
        current_embeddings = get_embeddings(self.embedding_encoder, current_dataloader)
        reference_embeddings_df = convert_tensor_to_df(reference_embeddings, "col_")
        current_embeddings_df = convert_tensor_to_df(current_embeddings, "col_")

        drift_results = {
            **self.evidently_detector.detect_drift(reference_embeddings_df, current_embeddings_df, is_warmup),
            **self.alibi_detector.detect_drift(reference_embeddings, current_embeddings, is_warmup),
        }

        # make the final decisions with the decision engines
        for metric_name, metric_result in drift_results.items():
            # to be able to evaluate multiple distance metrics on the same embeddings (without recalculating them
            # in another drift trigger), we support aggregation of different metrics within one DriftTrigger
            distance = (
                metric_result.distance
                if isinstance(metric_result.distance, float)
                else float(mean(metric_result.distance))
            )
            # overwrite the raw decision from the metric that is not of interest to us.
            drift_results[metric_name].is_drift = self.decision_policies[metric_name].evaluate_decision(distance)

        logger.info(f"[DataDriftDetector][Dataset {self.dataloader_info.dataset_id}]" + f"[Result] {drift_results}")
        if is_warmup:
            return False, {}

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

    def _any_metric_needs_calibration(self) -> bool:
        return any(metric.decision_criterion.needs_calibration for metric in self.config.metrics.values())


def _setup_detection_window_manager(
    windowing_strategy: DriftWindowingStrategy,
) -> DetectionWindows:
    if isinstance(windowing_strategy, AmountWindowingStrategy):
        return AmountDetectionWindows(windowing_strategy)
    if isinstance(windowing_strategy, TimeWindowingStrategy):
        return TimeDetectionWindows(windowing_strategy)
    raise ValueError(f"Unsupported windowing strategy: {windowing_strategy}")


def _setup_decision_engines(
    config: DataDriftTriggerConfig,
) -> dict[str, DriftDecisionPolicy]:
    decision_engines: dict[str, DriftDecisionPolicy] = {}
    for metric_name, metric_config in config.metrics.items():
        criterion = metric_config.decision_criterion
        assert (
            metric_config.num_permutations is None
        ), "Modyn doesn't allow hypothesis testing, it doesn't work in our context"
        if isinstance(criterion, ThresholdDecisionCriterion):
            decision_engines[metric_name] = ThresholdDecisionPolicy(config)
        elif isinstance(criterion, DynamicDecisionPolicy):
            decision_engines[metric_name] = DynamicDecisionPolicy(config)
    return decision_engines

from __future__ import annotations

import gc
import logging

from typing_extensions import override

from modyn.config.schema.pipeline import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.drift.criterion import (
    DynamicQuantileThresholdCriterion,
    DynamicRollingAverageThresholdCriterion,
    ThresholdDecisionCriterion,
)
from modyn.config.schema.pipeline.trigger.drift.detection_window import (
    AmountWindowingStrategy,
    DriftWindowingStrategy,
    TimeWindowingStrategy,
)
from modyn.config.schema.pipeline.trigger.drift.result import MetricResult
from modyn.supervisor.internal.triggers.batchedtrigger import BatchedTrigger
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
from modyn.supervisor.internal.triggers.drift.embedding.embeddings import get_embeddings
from modyn.supervisor.internal.triggers.drift.utils import convert_tensor_to_df
from modyn.supervisor.internal.triggers.trigger import TriggerContext
from modyn.supervisor.internal.triggers.utils.datasets.dataloader_info import (
    DataLoaderInfo,
)
from modyn.supervisor.internal.triggers.utils.datasets.prepare_dataloader import (
    prepare_trigger_dataloader_fixed_keys,
)
from modyn.supervisor.internal.triggers.utils.decision_policy import (
    DecisionPolicy,
    DynamicQuantileThresholdPolicy,
    DynamicRollingAverageThresholdPolicy,
    StaticThresholdDecisionPolicy,
)
from modyn.supervisor.internal.triggers.utils.model.downloader import ModelDownloader
from modyn.supervisor.internal.triggers.utils.model.stateful_model import StatefulModel
from modyn.supervisor.internal.triggers.utils.models import (
    DriftTriggerEvalLog,
    TriggerPolicyEvaluationLog,
)

logger = logging.getLogger(__name__)


class DataDriftTrigger(BatchedTrigger):
    """Triggers when a we detect drift in the embedding space of a dataset."""

    def __init__(self, config: DataDriftTriggerConfig):
        super().__init__(config)
        self.config = config
        self.context: TriggerContext | None = None

        self.most_recent_model_id: int | None = None
        self.model_refresh_needed: bool = False

        self.dataloader_info: DataLoaderInfo | None = None
        self.model_downloader: ModelDownloader | None = None
        self.model: StatefulModel | None = None

        self.windows = _setup_detection_windows(config.windowing_strategy)
        self._triggered_once = False

        self.evidently_detector = EvidentlyDriftDetector(config.metrics)
        self.alibi_detector = AlibiDriftDetector(config.metrics)

        # Every decision policy wraps one metric and is responsible for making decisions based on the metric's results
        # and the metric's range of distance values
        self.decision_policies = _setup_decision_policies(config)

        # list of reference windows for each warmup interval
        self.warmup_intervals: list[list[tuple[int, int]]] = []

    @override
    def init_trigger(self, context: TriggerContext) -> None:
        self.context = context
        self._init_dataloader_info()
        self._init_model_downloader()

    @override
    def _evaluate_batch(
        self,
        batch: list[tuple[int, int]],
        trigger_candidate_idx: int,
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> bool:
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

        Returns:
            Whether this batch resulted in a trigger.
        """
        self.windows.inform_data(batch)
        warmup_completed_prior_to_batch = self.warmup_trigger.completed

        # --------------------------------------------- Trigger Decision --------------------------------------------- #

        if (not self._triggered_once) or not warmup_completed_prior_to_batch:
            # storing the reference window for later calibration with the drift policy.
            drift_results: dict[str, MetricResult] = {}
            if not warmup_completed_prior_to_batch:
                # for the first detection the reference window is empty, therefore adding the current window
                self.warmup_intervals.append(
                    list(self.windows.reference if self._triggered_once else self.windows.current)
                )

            # delegate to the warmup policy
            delegated_trigger_results = self.warmup_trigger.delegate_inform(batch)
            triggered = not self._triggered_once or delegated_trigger_results

            self._triggered_once = True

            # For the last warmup interval we run all the warmup detections.
            # This allows to inform the metrics that use decision criteria
            # with calibration requirements about the warmup intervals.
            # They can then calibrate their thresholds.
            first_non_warmup = self.warmup_trigger.completed and not warmup_completed_prior_to_batch
            if first_non_warmup:
                # we can ignore the results as the decision criteria will keep track of the warmup results
                # internally
                if self._any_metric_needs_calibration():
                    for warmup_interval in self.warmup_intervals:
                        # we generate the calibration with different reference windows, the latest model and
                        # the current window
                        _warmup_triggered, _warmup_results = self._run_detection(
                            warmup_interval,
                            list(self.windows.current),
                            is_warmup=True,
                        )
                        if log:
                            warmup_log = DriftTriggerEvalLog(
                                triggered=_warmup_triggered,
                                trigger_index=-1,
                                evaluation_interval=(batch[0][1], batch[-1][1]),
                                detection_interval=(
                                    self.windows.current[0][1],
                                    self.windows.current[-1][1],
                                ),
                                reference_interval=(
                                    self.windows.reference[0][1],
                                    self.windows.reference[-1][1],
                                ),
                                drift_results=_warmup_results,
                            )
                            log.evaluations.append(warmup_log)

                # free the memory, but keep filled
                self.warmup_intervals = []
                gc.collect()

        else:
            # Run the detection
            triggered, drift_results = self._run_detection(
                list(self.windows.reference),
                list(self.windows.current),
                is_warmup=False,
            )

        # -------------------------------------------------- Log ------------------------------------------------- #

        drift_eval_log = DriftTriggerEvalLog(
            triggered=triggered,
            trigger_index=-1,
            evaluation_interval=(batch[0][1], batch[-1][1]),
            detection_interval=(
                self.windows.current[0][1],
                self.windows.current[-1][1],
            ),
            reference_interval=(
                (self.windows.reference[0][1], self.windows.reference[-1][1]) if self.windows.reference else (-1, -1)
            ),
            drift_results=drift_results,
        )

        if log:
            log.evaluations.append(drift_eval_log)

        if not self.warmup_trigger.completed:
            # During the warmup phase we want to allow the windows to reset themselves as if we detected drift;
            # If the window current window is cleared or not is eventually left to the windowing strategy.
            # We can call `inform_trigger` here and again in `inform_new_model` if `trigger=True`
            # as the call is idempotent.
            self.windows.inform_trigger()

        return triggered

    @override
    def inform_new_model(
        self,
        most_recent_model_id: int,
        number_samples: int | None = None,
        training_time: float | None = None,
    ) -> None:
        self.most_recent_model_id = most_recent_model_id
        self.model_refresh_needed = True
        self.windows.inform_trigger()

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     INTERNAL                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    def _run_detection(
        self,
        reference: list[tuple[int, int]],
        current: list[tuple[int, int]],
        is_warmup: bool,
    ) -> tuple[bool, dict[str, MetricResult]]:
        """Compare current data against reference data.

        current data: all untriggered samples in the sliding window in inform().
        reference data: the training samples of the previous trigger.
        Get the dataloaders, download the stateful model if necessary,
        compute embeddings of current and reference data, then run detection on the embeddings.
        """
        assert self.most_recent_model_id is not None
        assert self.dataloader_info is not None
        assert self.model_downloader is not None
        assert self.context and self.context.pipeline_config is not None
        assert len(reference) > 0
        assert len(current) > 0

        reference_dataloader = prepare_trigger_dataloader_fixed_keys(
            self.dataloader_info,
            [key for key, _ in reference],
            sample_size=self.config.sample_size,
        )

        current_dataloader = prepare_trigger_dataloader_fixed_keys(
            self.dataloader_info,
            [key for key, _ in current],
            sample_size=self.config.sample_size,
        )

        # Download most recent model as stateful model
        # TODO(417) Support custom model as stateful model
        if self.model_refresh_needed:
            self.model = self.model_downloader.setup_manager(
                self.most_recent_model_id, self.context.pipeline_config.training.device
            )
            self.model_refresh_needed = False

        # Compute embeddings
        assert self.model is not None

        # TODO(@robinholzi): reuse the embeddings as long as the reference window is not updated
        reference_embeddings = get_embeddings(self.model, reference_dataloader)
        current_embeddings = get_embeddings(self.model, current_dataloader)
        reference_embeddings_df = convert_tensor_to_df(reference_embeddings, "col_")
        current_embeddings_df = convert_tensor_to_df(current_embeddings, "col_")

        drift_results = {
            **self.evidently_detector.detect_drift(reference_embeddings_df, current_embeddings_df, is_warmup),
            **self.alibi_detector.detect_drift(reference_embeddings, current_embeddings, is_warmup),
        }

        # make the final decisions with the decision policies
        for metric_name, metric_result in drift_results.items():
            # overwrite the raw decision from the metric that is not of interest to us.
            drift_results[metric_name].is_drift = self.decision_policies[metric_name].evaluate_decision(
                metric_result.distance
            )

        logger.info("[DataDriftDetector][Dataset %s][Result] %s", self.dataloader_info.dataset_id, drift_results)
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

    def _init_model_downloader(self) -> None:
        assert self.context is not None

        self.model_downloader = ModelDownloader(
            self.context.modyn_config,
            self.context.pipeline_id,
            self.context.base_dir,
            f"{self.context.modyn_config.modyn_model_storage.address}",
        )

    def _any_metric_needs_calibration(self) -> bool:
        return any(metric.decision_criterion.needs_calibration for metric in self.config.metrics.values())


def _setup_detection_windows(
    windowing_strategy: DriftWindowingStrategy,
) -> DetectionWindows:
    if isinstance(windowing_strategy, AmountWindowingStrategy):
        return AmountDetectionWindows(windowing_strategy)
    if isinstance(windowing_strategy, TimeWindowingStrategy):
        return TimeDetectionWindows(windowing_strategy)
    raise ValueError(f"Unsupported windowing strategy: {windowing_strategy}")


def _setup_decision_policies(
    config: DataDriftTriggerConfig,
) -> dict[str, DecisionPolicy]:
    policies: dict[str, DecisionPolicy] = {}
    for metric_name, metric_config in config.metrics.items():
        criterion = metric_config.decision_criterion
        assert (
            metric_config.num_permutations is None
        ), "Modyn doesn't allow hypothesis testing, it doesn't work in our context"
        if isinstance(criterion, ThresholdDecisionCriterion):
            policies[metric_name] = StaticThresholdDecisionPolicy(
                threshold=criterion.threshold, triggering_direction="higher"
            )
        elif isinstance(criterion, DynamicQuantileThresholdCriterion):
            policies[metric_name] = DynamicQuantileThresholdPolicy(
                window_size=criterion.window_size,
                quantile=criterion.quantile,
                triggering_direction="higher",
            )
        elif isinstance(criterion, DynamicRollingAverageThresholdCriterion):
            policies[metric_name] = DynamicRollingAverageThresholdPolicy(
                window_size=criterion.window_size,
                deviation=criterion.deviation,
                absolute=criterion.absolute,
                triggering_direction="higher",
            )
    return policies

from __future__ import annotations

import logging
from collections.abc import Generator

from modyn.config.schema.pipeline.trigger.performance.criterion import (
    DynamicPerformanceThresholdCriterion,
    StaticNumberAvoidableMisclassificationCriterion,
    StaticPerformanceThresholdCriterion,
)
from modyn.config.schema.pipeline.trigger.performance.performance import (
    PerformanceTriggerConfig,
)
from modyn.evaluator.internal.core_evaluation import perform_evaluation, setup_metrics
from modyn.supervisor.internal.triggers.models import (
    PerformanceTriggerEvalLog,
    TriggerPolicyEvaluationLog,
)
from modyn.supervisor.internal.triggers.performance.data_density_tracker import (
    DataDensityTracker,
)
from modyn.supervisor.internal.triggers.performance.decision_policy import (
    DynamicPerformanceThresholdDecisionPolicy,
    PerformanceDecisionPolicy,
    StaticNumberAvoidableMisclassificationDecisionPolicy,
    StaticPerformanceThresholdDecisionPolicy,
)
from modyn.supervisor.internal.triggers.performance.performance_tracker import (
    PerformanceTracker,
)
from modyn.supervisor.internal.triggers.trigger import Trigger, TriggerContext
from modyn.supervisor.internal.triggers.utils.datasets.dataloader_info import (
    DataLoaderInfo,
)
from modyn.supervisor.internal.triggers.utils.datasets.prepare_dataloader import (
    prepare_trigger_dataloader_fixed_keys,
)
from modyn.supervisor.internal.triggers.utils.model.downloader import ModelDownloader
from modyn.supervisor.internal.triggers.utils.model.manager import ModelManager
from modyn.utils.utils import LABEL_TRANSFORMER_FUNC_NAME, deserialize_function

logger = logging.getLogger(__name__)


class PerformanceTrigger(Trigger):
    """Trigger based on the performance of the model.

    We support a simple performance drift approach that compares the
    most recent model performance with an expected performance value
    that can be static or dynamic through a rolling average.

    Additionally we support a regret based approach where the number of
    avoidable misclassifications (misclassifications that could have
    been avoided if we would have triggered) is compared to a threshold.
    """

    def __init__(self, config: PerformanceTriggerConfig) -> None:
        super().__init__()

        self.config = config
        self.context: TriggerContext | None = None
        self.previous_model_id: int | None = None

        self.dataloader_info: DataLoaderInfo | None = None
        self.model_downloader: ModelDownloader | None = None
        self.model_manager: ModelManager | None = None

        # allows to detect drift in a fixed interval
        self._sample_left_until_detection = config.detection_interval_data_points

        self.data_density = DataDensityTracker(config.data_density_window_size)
        self.performance_tracker = PerformanceTracker(config.performance_triggers_window_size)

        self.decision_policies = _setup_decision_policies(config)

        self._triggered_once = False
        self.model_refresh_needed = False
        self._metrics = setup_metrics(config.evaluation.dataset.metrics)
        self._last_data_interval: list[tuple[int, int]] = []

        self._label_transformer_function = deserialize_function(
            config.evaluation.label_transformer_function, LABEL_TRANSFORMER_FUNC_NAME
        )

    def init_trigger(self, context: TriggerContext) -> None:
        self.context = context
        self._init_dataloader_info()
        self._init_model_downloader()

    def inform(
        self,
        new_data: list[tuple[int, int, int]],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> Generator[int, None, None]:
        new_key_ts = [(key, timestamp) for key, timestamp, _ in new_data]

        # index of the first unprocessed data point in the batch
        processing_head_in_batch = 0

        # Go through remaining data in new data in batches of `detect_interval`
        while True:
            if self._sample_left_until_detection - len(new_key_ts) > 0:
                # No detection in this trigger because of too few data points to fill detection interval
                self._sample_left_until_detection -= len(new_key_ts)
                return

            # At least one detection, fill up window up to that detection
            next_detection_interval = new_key_ts[: self._sample_left_until_detection]
            self.data_density.inform_data(next_detection_interval)

            # Update the remaining data
            processing_head_in_batch += len(next_detection_interval)
            new_key_ts = new_key_ts[len(next_detection_interval) :]

            # Reset for next detection
            self._sample_left_until_detection = self.config.detection_interval_data_points

            # Run the evaluation (even if we don't use the result, e.g. for the first forced trigger)
            self.data_density.inform_data(new_key_ts)

            num_samples, num_misclassifications, evaluation_scores = self._run_evaluation(interval_data=new_key_ts)

            self.performance_tracker.inform_evaluation(
                num_samples=num_samples,
                num_misclassifications=num_misclassifications,
                evaluation_scores=evaluation_scores,
            )

            policy_decisions: dict[str, bool] = {}

            # The first ever detection will always trigger
            if not self._triggered_once:
                # If we've never triggered before, always trigger
                self._triggered_once = True
                triggered = True

                # in the first interval we don't evaluate the decision policies as they might require one trigger
                # to have happened before in order to derive forecasts

            else:
                # evaluate the decision policies
                any_policy_triggered = False
                for policy_name, policy in self.decision_policies.items():
                    policy_decisions[policy_name] = policy.evaluate_decision(
                        update_interval=self.config.detection_interval_data_points,
                        evaluation_scores=evaluation_scores,
                        data_density=self.data_density,
                        performance_tracker=self.performance_tracker,
                        mode=self.config.mode,
                        method=self.config.forecasting_method,
                    )
                    if policy_decisions[policy_name]:
                        any_policy_triggered = True

                triggered = any_policy_triggered

            if triggered:
                for policy in self.decision_policies.values():
                    policy.inform_trigger()  # resets the internal state (e.g. misclassification counters)

            trigger_idx = processing_head_in_batch - 1

            drift_eval_log = PerformanceTriggerEvalLog(
                triggered=triggered,
                trigger_index=trigger_idx,
                evaluation_interval=(new_key_ts[0][1], new_key_ts[-1][1]),
                num_samples=num_samples,
                num_misclassifications=num_misclassifications,
                evaluation_scores=evaluation_scores,
                policy_decisions=policy_decisions,
            )
            if log:
                log.evaluations.append(drift_eval_log)

            if triggered:
                self._last_data_interval = new_key_ts
                yield trigger_idx

    def inform_new_model(self, previous_model_id: int) -> None:
        self.previous_model_id = previous_model_id
        self.model_refresh_needed = True

        # Perform an evaluation of the NEW model on the last evaluation interval, we will derive expected performance
        # forecasts from these evaluations.
        num_samples, num_misclassifications, evaluation_scores = self._run_evaluation(
            interval_data=self._last_data_interval
        )

        self.performance_tracker.inform_trigger(
            num_samples=num_samples,
            num_misclassifications=num_misclassifications,
            evaluation_scores=evaluation_scores,
        )

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     Internal                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    def _run_evaluation(self, interval_data: list[tuple[int, int]]) -> tuple[int, int, dict[str, float]]:
        """Run the evaluation on the given interval data."""
        assert self.previous_model_id is not None
        assert self.dataloader_info is not None
        assert self.model_downloader is not None
        assert self.context and self.context.pipeline_config is not None

        evaluation_dataloader = prepare_trigger_dataloader_fixed_keys(
            self.dataloader_info, [key for key, _ in interval_data]
        )

        # Download previous model as model manager
        if self.model_refresh_needed:
            self.model_manager = self.model_downloader.setup_manager(
                self.previous_model_id, self.context.pipeline_config.training.device
            )
            self.model_refresh_needed = False

        # Run evaluation
        assert self.model_manager is not None
        num_samples, eval_results = perform_evaluation(
            model=self.model_manager,
            dataloader=evaluation_dataloader,
            device=self.config.evaluation.device,
            metrics=self._metrics,
            label_transformer_function=self._label_transformer_function,
            amp=False,
        )

        evaluation_scores = {metric.get_name(): metric.get_evaluation_result() for metric in self._metrics}

        num_misclassifications = 0  # TODO

        # TODO

        return (num_samples, num_misclassifications, evaluation_scores)

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


def _setup_decision_policies(
    config: PerformanceTriggerConfig,
) -> dict[str, PerformanceDecisionPolicy]:
    """Policy factory that creates the decision policies based on the given
    configuration."""
    policies: dict[str, PerformanceDecisionPolicy] = {}
    for name, criterion in config.decision_criteria.items():
        if isinstance(criterion, StaticPerformanceThresholdCriterion):
            policies[name] = StaticPerformanceThresholdDecisionPolicy(criterion)
        elif isinstance(criterion, DynamicPerformanceThresholdCriterion):
            policies[name] = DynamicPerformanceThresholdDecisionPolicy(criterion)
        elif isinstance(criterion, StaticNumberAvoidableMisclassificationCriterion):
            policies[name] = StaticNumberAvoidableMisclassificationDecisionPolicy(criterion)
    return policies

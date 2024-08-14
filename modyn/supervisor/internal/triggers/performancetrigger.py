from __future__ import annotations

import logging
from collections.abc import Generator

from modyn.config.schema.pipeline.trigger.performance.performance import (
    PerformanceTriggerConfig,
)
from modyn.evaluator.internal.core_evaluation import perform_evaluation, setup_metrics
from modyn.supervisor.internal.triggers.drift.embedding.model.downloader import (
    ModelDownloader,
)
from modyn.supervisor.internal.triggers.drift.embedding.model.manager import (
    ModelManager,
)
from modyn.supervisor.internal.triggers.models import TriggerPolicyEvaluationLog
from modyn.supervisor.internal.triggers.performance.data_density import (
    DataDensityTracker,
)
from modyn.supervisor.internal.triggers.performance.performance import (
    PerformanceTracker,
)
from modyn.supervisor.internal.triggers.trigger import Trigger, TriggerContext
from modyn.supervisor.internal.triggers.trigger_datasets.dataloader_info import (
    DataLoaderInfo,
)
from modyn.supervisor.internal.triggers.trigger_datasets.prepare_dataloader import (
    prepare_trigger_dataloader_fixed_keys,
)

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

        self._sample_left_until_detection = (
            config.detection_interval_data_points
        )  # allows to detect drift in a fixed interval

        self.data_density = DataDensityTracker(config.data_density_window_size)
        self.performance_tracker = PerformanceTracker(config.performance_triggers_window_size)

        self._triggered_once = False
        self._metrics = setup_metrics(config.evaluation.dataset.metrics)

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

            # The first ever detection will always trigger
            if not self._triggered_once:
                # If we've never triggered before, always trigger
                self._triggered_once = True
                triggered = True

            else:
                # Run the detection
                # TODO: inform performance_tracker before policy call
                triggered = self._run_detection(new_key_ts)

            if triggered:
                yield 1
                # TODO
                # trigger_idx = processing_head_in_batch - 1
                # yield from self._handle_drift_result(
                #     triggered, trigger_idx, drift_results, log=log
                # )

            # TODO: dicison policy needs inform of trigger

    def inform_previous_model(self, previous_model_id: int) -> None:
        self.previous_model_id = previous_model_id
        self.model_updated = True

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     Internal                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    def _run_detection(self, interval_data: list[tuple[int, int]]) -> bool:
        """Compare current data against reference data.

        current data: all untriggered samples in the sliding window in inform().
        reference data: the training samples of the previous trigger.
        Get the dataloaders, download the embedding encoder model if necessary,
        compute embeddings of current and reference data, then run detection on the embeddings.
        """
        assert self.previous_model_id is not None
        assert self.dataloader_info is not None
        assert self.model_downloader is not None
        assert self.context and self.context.pipeline_config is not None

        evaluation_dataloader = prepare_trigger_dataloader_fixed_keys(
            self.dataloader_info, [key for key, _ in interval_data]
        )

        # Download previous model as embedding encoder
        if self.model_updated:
            self.model_manager = self.model_downloader.setup_manager(
                self.previous_model_id, self.context.pipeline_config.training.device
            )
            self.model_updated = False

        # Run evaluation
        assert self.model_manager is not None
        num_samples, eval_results = perform_evaluation(
            model=self.model_manager,
            dataloader=evaluation_dataloader,
            device=self.config.evaluation.device,
            metrics=self._metrics,
            label_transformer_function=self.config.evaluation.label_transformer_function,
            amp=False,  # TODO?
        )

        return False  # TODO:

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


# TODO: log which of the criterions led to the trigger

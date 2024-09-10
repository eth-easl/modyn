from modyn.config.schema.pipeline.trigger.performance.performance import (
    _InternalPerformanceTriggerConfig,
)
from modyn.evaluator.internal.core_evaluation import perform_evaluation, setup_metrics
from modyn.evaluator.internal.metrics.accuracy import Accuracy
from modyn.supervisor.internal.triggers.performance.data_density_tracker import (
    DataDensityTracker,
)
from modyn.supervisor.internal.triggers.performance.performance_tracker import (
    PerformanceTracker,
)
from modyn.supervisor.internal.triggers.trigger import TriggerContext
from modyn.supervisor.internal.triggers.utils.datasets.dataloader_info import (
    DataLoaderInfo,
)
from modyn.supervisor.internal.triggers.utils.datasets.prepare_dataloader import (
    prepare_trigger_dataloader_fixed_keys,
)
from modyn.supervisor.internal.triggers.utils.model.downloader import ModelDownloader
from modyn.supervisor.internal.triggers.utils.model.stateful_model import StatefulModel
from modyn.utils.utils import LABEL_TRANSFORMER_FUNC_NAME, deserialize_function


class PerformanceTriggerMixin:
    """Mixin that provides internal functionality for performance triggers but
    not the trigger policy itself."""

    def __init__(self, config: _InternalPerformanceTriggerConfig) -> None:
        self.config = config
        self.context: TriggerContext | None = None

        self.data_density = DataDensityTracker(config.data_density_window_size)
        self.performance_tracker = PerformanceTracker(config.performance_triggers_window_size)

        self.model_refresh_needed = False
        self.most_recent_model_id: int | None = None
        self.dataloader_info: DataLoaderInfo | None = None
        self.model_downloader: ModelDownloader | None = None
        self.sf_model: StatefulModel | None = None

        self._metrics = setup_metrics(config.evaluation.dataset.metrics)

        self._label_transformer_function = (
            deserialize_function(
                config.evaluation.label_transformer_function,
                LABEL_TRANSFORMER_FUNC_NAME,
            )
            if config.evaluation.label_transformer_function
            else None
        )

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     Internal                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    def _init_trigger(self, context: TriggerContext) -> None:
        self.context = context
        self._init_dataloader_info()
        self._init_model_downloader()

    def _inform_new_model(self, most_recent_model_id: int, last_detection_interval: list[tuple[int, int]]) -> None:
        """Needs to called by the subclass's inform_new_model method."""

        self.most_recent_model_id = most_recent_model_id
        self.model_refresh_needed = True

        assert last_detection_interval

        # Perform an evaluation of the NEW model on the last evaluation interval, we will derive expected performance
        # forecasts from these evaluations.
        # num_samples, num_misclassifications, evaluation_scores = self._run_evaluation(
        #     interval_data=last_detection_interval
        # )
        assert self.config.mode == "hindsight", (
            "Forecasting mode is not supported yet, it requires tracking the performance right after trigger. "
            "However, after triggers the models has learned from the last detection interval. We would need to "
            "maintain a holdout set for this."
        )

        # self.performance_tracker.inform_trigger(
        #     num_samples=num_samples,
        #     num_misclassifications=num_misclassifications,
        #     evaluation_scores=evaluation_scores,
        # )

    def _run_evaluation(
        self,
        interval_data: list[tuple[int, int]],
    ) -> tuple[int, int, int, dict[str, float]]:  # pragma: no cover
        """Run the evaluation on the given interval data."""
        assert self.most_recent_model_id is not None
        assert self.dataloader_info is not None
        assert self.model_downloader is not None
        assert self.context and self.context.pipeline_config is not None

        # Since the metric objects are stateful, we need to re-instantiate them before each evaluation.
        self._metrics = setup_metrics(self.config.evaluation.dataset.metrics)

        evaluation_dataloader = prepare_trigger_dataloader_fixed_keys(
            self.dataloader_info, [key for key, _ in interval_data]
        )

        # Download most recent model as stateful model
        if self.model_refresh_needed:
            self.sf_model = self.model_downloader.setup_manager(
                self.most_recent_model_id, self.context.pipeline_config.training.device
            )
            self.model_refresh_needed = False

        # Run evaluation
        assert self.sf_model is not None

        eval_results = perform_evaluation(
            model=self.sf_model.model.model,
            dataloader=evaluation_dataloader,
            device=self.config.evaluation.device,
            metrics=self._metrics,
            label_transformer_function=self._label_transformer_function,
            amp=False,
        )

        evaluation_scores = {
            metric_name: metric.get_evaluation_result() for metric_name, metric in self._metrics.items()
        }

        accuracy_metric = eval_results.metrics_data["Accuracy"]
        assert isinstance(accuracy_metric, Accuracy)
        num_misclassifications = accuracy_metric.samples_seen - accuracy_metric.total_correct

        return (
            self.most_recent_model_id,
            eval_results.num_samples,
            num_misclassifications,
            evaluation_scores,
        )

    def _init_dataloader_info(self) -> None:
        assert self.context

        training_config = self.context.pipeline_config.training
        data_config = self.context.pipeline_config.data

        self.dataloader_info = DataLoaderInfo(
            pipeline_id=self.context.pipeline_id,
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

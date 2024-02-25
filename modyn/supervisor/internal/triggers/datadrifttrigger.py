import logging
import pathlib
from collections.abc import Generator
from typing import Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import EmbeddingsDriftMetric
from evidently.metrics.data_drift.embedding_drift_methods import distance, mmd, model, ratio
from evidently.report import Report

# pylint: disable-next=no-name-in-module
from modyn.supervisor.internal.triggers.model_wrappers import ModynModelWrapper
from modyn.supervisor.internal.triggers.trigger import Trigger
from modyn.supervisor.internal.triggers.trigger_datasets import DataLoaderInfo
from modyn.supervisor.internal.triggers.utils import (
    prepare_trigger_dataloader_by_trigger,
    prepare_trigger_dataloader_given_keys,
)

# import PIL


logger = logging.getLogger(__name__)


class DataDriftTrigger(Trigger):
    """Triggers when a certain number of data points have been used."""

    def __init__(self, trigger_config: dict):
        self.pipeline_id: Optional[int] = None
        self.pipeline_config: Optional[dict] = None
        self.modyn_config: Optional[dict] = None
        self.base_dir: Optional[pathlib.Path] = None
        self.model_wrapper: Optional[ModynModelWrapper] = None
        self.dataloader_info: Optional[DataLoaderInfo] = None

        self.previous_trigger_id: Optional[int] = None
        self.previous_data_points: Optional[int] = None
        self.previous_model_id: Optional[int] = None
        self.model_updated: bool = False

        self.detection_data_points: int = 2
        if "data_points_for_detection" in trigger_config.keys():
            self.detection_data_points = trigger_config["data_points_for_detection"]
        assert self.detection_data_points > 0, "data_points_for_trigger needs to be at least 1"

        self.drift_threshold: float = 0.55
        if "drift_threshold" in trigger_config.keys():
            self.drift_threshold = trigger_config["drift_threshold"]
        assert self.drift_threshold >= 0 and self.drift_threshold <= 1, "drift_threshold range [0,1]"

        self.sample_size: Optional[int] = None
        if "sample_size" in trigger_config.keys():
            self.sample_size = trigger_config["sample_size"]
        assert self.sample_size is None or self.sample_size > 0, "sample_size needs to be at least 1"

        self.metrics = [
            EmbeddingsDriftMetric(
                "data",
                drift_method=model(
                    threshold=self.drift_threshold,
                    bootstrap=None,
                    quantile_probability=0.95,
                    pca_components=None,
                ),
            ),
            EmbeddingsDriftMetric(
                "data",
                drift_method=ratio(
                    component_stattest="wasserstein",
                    component_stattest_threshold=0.1,
                    threshold=0.2,
                    pca_components=None,
                ),
            ),
            EmbeddingsDriftMetric(
                "data",
                drift_method=distance(
                    dist="euclidean",  # "euclidean", "cosine", "cityblock" or "chebyshev"
                    threshold=0.2,
                    bootstrap=None,
                    quantile_probability=0.05,
                    pca_components=None,
                ),
            ),
            EmbeddingsDriftMetric(
                "data",
                drift_method=mmd(
                    threshold=0.015,
                    bootstrap=None,
                    quantile_probability=0.05,
                    pca_components=None,
                ),
            ),
        ]

        self.data_cache = []
        self.untriggered_detection_data_points = 0

        super().__init__(trigger_config)

    def run_detection(self, reference_embeddings_df: pd.DataFrame, current_embeddings_df: pd.DataFrame) -> bool:
        # Run Evidently detection
        column_mapping = ColumnMapping(embeddings={"data": reference_embeddings_df.columns})

        # https://docs.evidentlyai.com/user-guide/customization/embeddings-drift-parameters
        report = Report(metrics=self.metrics)
        report.run(
            reference_data=reference_embeddings_df, current_data=current_embeddings_df, column_mapping=column_mapping
        )
        result = report.as_dict()
        result_print = [(x["result"]["drift_score"], x["result"]["method_name"]) for x in result["metrics"]]
        logger.info(
            f"[DataDriftDetector][Prev Trigger {self.previous_trigger_id}][Dataset {self.dataloader_info.dataset_id}]"
            + f"[Result] {result_print}"
        )

        # with open(self.drift_dir / f"{self.previous_trigger_id}.json", "w") as f:
        #     json.dump(result, f)

        true_count = 0
        for metric in result["metrics"]:
            true_count += int(metric["result"]["drift_detected"])
        return true_count * 2 > len(result["metrics"])

    def detect_drift(self, idx_start, idx_end) -> bool:
        assert self.previous_trigger_id is not None
        assert self.previous_data_points is not None and self.previous_data_points > 0
        assert self.previous_model_id is not None
        assert self.dataloader_info is not None

        # get training data keys of previous trigger from storage
        logger.info(
            f"[Prev Trigger {self.previous_trigger_id}][Prev Model {self.previous_model_id}] Start drift detection"
        )
        reference_dataloader = prepare_trigger_dataloader_by_trigger(
            self.previous_trigger_id,
            self.dataloader_info,
            data_points_in_trigger=self.previous_data_points,
            sample_size=self.sample_size,
        )

        # get new data
        current_keys, timestamps, _ = zip(*self.data_cache[idx_start:idx_end])  # type: ignore
        num_per_t = {}
        for t in timestamps:
            num_per_t[t] = num_per_t.get(t, 0) + 1
        logger.debug(
            f"[DataDriftDetector][Prev Trigger {self.previous_trigger_id}][Dataset {self.dataloader_info.dataset_id}]"
            + f"[Detect Timestamps] {len(num_per_t)}:{num_per_t}"
        )
        current_dataloader = prepare_trigger_dataloader_given_keys(
            self.previous_trigger_id + 1,
            self.dataloader_info,
            current_keys,
            sample_size=self.sample_size,
        )

        # Fetch model used for embedding
        # TODO(JZ): custom embedding???
        if self.model_updated:
            self.model_wrapper.download(self.previous_model_id)
            self.model_updated = False

        # Compute embeddings
        reference_embeddings_df = self.model_wrapper.get_embeddings_evidently_format(reference_dataloader)
        current_embeddings_df = self.model_wrapper.get_embeddings_evidently_format(current_dataloader)
        # reference_embeddings_df.to_csv(f"{self.debug_dir}/{self.pipeline_id}_ref.csv", index=False)
        # current_embeddings_df.to_csv(f"{self.debug_dir}/{self.pipeline_id}_cur.csv", index=False)

        return self.run_detection(reference_embeddings_df, current_embeddings_df)
        # return True

    # the supervisor informs the Trigger about the previous trigger before it calls next()

    def inform(self, new_data: list[tuple[int, int, int]]) -> Generator[int]:
        # logger.debug(f"[DataDriftDetector][Prev Trigger {self.previous_trigger_id}]"
        #     + f"[Dataset {self.dataloader_info.dataset_id}]"
        #     + f"[Cached data] {len(self.data_cache)}, [New data] {len(new_data)}")

        # add new data to data_cache
        self.data_cache.extend(new_data)

        cached_data_points = len(self.data_cache)
        total_data_for_detection = len(self.data_cache)
        detection_data_idx_start = 0
        detection_data_idx_end = 0
        while total_data_for_detection >= self.detection_data_points:
            total_data_for_detection -= self.detection_data_points
            detection_data_idx_end += self.detection_data_points
            if detection_data_idx_end <= self.untriggered_detection_data_points:
                continue

            # trigger id doesn't always start from 0, but always increments by 1
            if self.previous_trigger_id is None:
                # if no previous model exists, always trigger
                triggered = True
            else:
                # if exist previous model, detect drift
                triggered = self.detect_drift(detection_data_idx_start, detection_data_idx_end)

            if triggered:
                trigger_data_points = detection_data_idx_end - detection_data_idx_start
                # logger.debug(f"[DataDriftDetector][Prev Trigger {self.previous_trigger_id}]"
                #   + f" new{len(new_data)} cache{cached_data_points} trigger{trigger_data_points}")
                trigger_idx = len(new_data) - (cached_data_points - trigger_data_points) - 1

                logger.debug(
                    f"[DataDriftDetector][Prev Trigger {self.previous_trigger_id}]"
                    + f"[Dataset {self.dataloader_info.dataset_id}]"
                    + f"[Trigger data points] {trigger_data_points}, [Trigger index] {trigger_idx}"
                )
                _, timestamps, _ = zip(
                    *self.data_cache[detection_data_idx_start:detection_data_idx_end]
                )  # type: ignore
                num_per_t = {}
                for t in timestamps:
                    num_per_t[t] = num_per_t.get(t, 0) + 1
                logger.debug(
                    f"[DataDriftDetector][Prev Trigger {self.previous_trigger_id}]"
                    + f"[Dataset {self.dataloader_info.dataset_id}]"
                    + f"[Trigger Timestamps] {len(num_per_t)}:{num_per_t}"
                )

                # update bookkeeping and pointers
                cached_data_points -= trigger_data_points
                detection_data_idx_start = detection_data_idx_end
                yield trigger_idx

        # remove triggered data
        self.data_cache = self.data_cache[detection_data_idx_start:]
        self.untriggered_detection_data_points = detection_data_idx_end - detection_data_idx_start

    def _init_dataloader_info(self) -> None:
        assert self.pipeline_id is not None
        assert self.pipeline_config is not None
        assert self.modyn_config is not None
        assert self.base_dir is not None

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

    def _init_model_wrapper(self):
        assert self.pipeline_id is not None
        assert self.pipeline_config is not None
        assert self.modyn_config is not None
        assert self.base_dir is not None

        self.model_wrapper = ModynModelWrapper(
            self.modyn_config,
            self.pipeline_id,
            self.pipeline_config["training"]["device"],
            self.base_dir,
            f"{self.modyn_config['model_storage']['hostname']}:{self.modyn_config['model_storage']['port']}",
        )

    def _create_dirs(self):
        assert self.pipeline_id is not None
        assert self.base_dir is not None

        # self.exp_output_dir = self.base_dir / f"{self.dataloader_info.dataset_id}_{self.pipeline_id}"
        # self.drift_dir = self.exp_output_dir / "drift"
        # os.makedirs(self.drift_dir, exist_ok=True)
        # self.debug_dir = self.exp_output_dir / "debug"
        # os.makedirs(self.debug_dir, exist_ok=True)

    def init_data_drift_trigger(
        self, pipeline_id: int, pipeline_config: dict, modyn_config: dict, base_dir: pathlib.Path
    ):
        self.pipeline_id = pipeline_id
        self.pipeline_config = pipeline_config
        self.modyn_config = modyn_config
        self.base_dir = base_dir

        self._init_dataloader_info()
        self._init_model_wrapper()
        self._create_dirs()

    def inform_previous_trigger_and_model(
        self, previous_trigger_id: Optional[int], previous_model_id: Optional[int]
    ) -> None:
        self.previous_trigger_id = previous_trigger_id
        self.previous_model_id = previous_model_id
        if previous_model_id is not None:
            self.model_updated = True

    def inform_previous_trigger_data_points(self, previous_trigger_id: int, data_points: int) -> None:
        assert self.previous_trigger_id == previous_trigger_id
        self.previous_data_points = data_points

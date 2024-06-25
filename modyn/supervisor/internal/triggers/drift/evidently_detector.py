from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from evidently import ColumnMapping
from evidently.metrics import EmbeddingsDriftMetric
from evidently.metrics.data_drift import embedding_drift_methods
from evidently.report import Report
from modyn.config.schema.pipeline import EvidentlyDriftMetric, MetricResult
from typing_extensions import override

from .drift_detector import DriftDetector

logger = logging.getLogger(__name__)

EVIDENTLY_COLUMN_MAPPING_NAME = "data"


class EvidentlyDriftDetector(DriftDetector):
    def __init__(self, metrics_config: dict[str, EvidentlyDriftMetric]):
        super().__init__(metrics_config)
        self.evidently_metrics = _get_evidently_metrics(metrics_config)

    @override
    def init_detector(self) -> None:
        pass

    @override
    def detect_drift(
        self,
        embeddings_ref: pd.DataFrame | np.ndarray | torch.Tensor,
        embeddings_cur: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> dict[str, MetricResult]:
        assert isinstance(embeddings_ref, pd.DataFrame)
        assert isinstance(embeddings_cur, pd.DataFrame)

        # Run Evidently detection
        # ColumnMapping is {mapping name: column indices},
        # an Evidently way of identifying (sub)columns to use in the detection.
        # e.g. {"even columns": [0,2,4]}.
        column_mapping = ColumnMapping(embeddings={EVIDENTLY_COLUMN_MAPPING_NAME: embeddings_ref.columns})

        # https://docs.evidentlyai.com/user-guide/customization/embeddings-drift-parameters
        report = Report(metrics=self.evidently_metrics)
        report.run(reference_data=embeddings_ref, current_data=embeddings_cur, column_mapping=column_mapping)
        results_raw = report.as_dict()

        results = {
            metric: MetricResult(
                metric_id=metric,
                is_drift=metric_result["result"]["drift_detected"],
                distance=metric_result["result"]["drift_score"],
            )
            for metric, metric_result in results_raw["metrics"].items()
        }
        return results


# -------------------------------------------------------------------------------------------------------------------- #
#                                                       Internal                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


def _get_evidently_metrics(metrics_config: dict[str, EvidentlyDriftMetric]) -> dict[str, EmbeddingsDriftMetric]:
    """This function instantiates an Evidently metric given metric configuration.
    If we want to support multiple metrics in the future, we can loop through the configurations.

    Evidently metric configurations follow exactly the four DriftMethods defined in embedding_drift_methods:
    model, distance, mmd, ratio
    If metric_name not given, we use the default 'model' metric.
    Otherwise, we use the metric given by metric_name, with optional metric configuration specific to the metric.
    """
    metrics = {
        metric_ref: EmbeddingsDriftMetric(EVIDENTLY_COLUMN_MAPPING_NAME, _evidently_metric_factory(config))
        for metric_ref, config in metrics_config.items()
    }
    return metrics


def _evidently_metric_factory(config: EvidentlyDriftMetric) -> EmbeddingsDriftMetric:
    if config.id == "EvidentlyMmdDriftMetric":
        return embedding_drift_methods.mmd(
            threshold=config.threshold,
            bootstrap=config.bootstrap,
            quantile_probability=config.quantile_probability,
            pca_components=config.num_pca_component,
        )
    if config.id == "EvidentlyModelDriftMetric":
        return embedding_drift_methods.model(
            threshold=config.threshold,
            bootstrap=config.bootstrap,
            quantile_probability=config.quantile_probability,
            pca_components=config.num_pca_component,
        )
    if config.id == "EvidentlyRatioDriftMetric":
        return embedding_drift_methods.ratio(
            component_stattest=config.component_stattest,
            component_stattest_threshold=config.component_stattest_threshold,
            threshold=config.threshold,
            pca_components=config.num_pca_component,
        )
    if config.id == "EvidentlySimpleDistanceDriftMetric":
        return embedding_drift_methods.distance(
            dist=config.distance_metric,
            threshold=config.threshold,
            bootstrap=config.bootstrap,
            pca_components=config.num_pca_component,
            quantile_probability=config.quantile_probability,
        )

    raise NotImplementedError(f"Metric {config.id} is not supported in EvidentlyDriftMetric.")

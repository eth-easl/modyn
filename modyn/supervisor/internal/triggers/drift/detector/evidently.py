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
from evidently.calculations.stattests.hellinger_distance import _hellinger_distance
from evidently.core import ColumnType
from .drift import DriftDetector

logger = logging.getLogger(__name__)

EVIDENTLY_COLUMN_MAPPING_NAME = "data"


class EvidentlyDriftDetector(DriftDetector):
    def __init__(self, metrics_config: dict[str, EvidentlyDriftMetric]):
        self.evidently_metrics_config = {
            metric_ref: config for metric_ref, config in metrics_config.items() if config.id.startswith("Evidently")
        }
        super().__init__(self.evidently_metrics_config)
        self.evidently_metrics = _get_evidently_metrics(self.evidently_metrics_config)

    def init_detector(self) -> None:
        pass

    def detect_drift(
        self,
        embeddings_ref: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor,
        embeddings_cur: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor,
        is_warmup: bool,
    ) -> dict[str, MetricResult]:
        if isinstance(embeddings_ref, torch.Tensor):
            embeddings_ref = embeddings_ref.cpu().numpy()

        if isinstance(embeddings_cur, torch.Tensor):
            embeddings_cur = embeddings_cur.cpu().numpy()

        if isinstance(embeddings_ref, np.ndarray):
            assert len(embeddings_ref.shape) == 2
            embeddings_ref = pd.DataFrame(embeddings_ref)

        if isinstance(embeddings_cur, np.ndarray):
            assert len(embeddings_cur.shape) == 2
            embeddings_cur = pd.DataFrame(embeddings_cur)

        # Run Evidently embedding drift detection
        # ColumnMapping is {mapping name: column indices},
        # an Evidently way of identifying (sub)columns to use in the detection.
        # e.g. {"even columns": [0,2,4]}.
        mapped_columns = list(map(str, embeddings_ref.columns))
        embeddings_ref.columns = mapped_columns
        embeddings_cur.columns = mapped_columns
        column_mapping = ColumnMapping(embeddings={EVIDENTLY_COLUMN_MAPPING_NAME: mapped_columns})

        # https://docs.evidentlyai.com/user-guide/customization/embeddings-drift-parameters
        report = Report(
            metrics=[
                self.evidently_metrics[name][1]
                for name in self.evidently_metrics
                if not is_warmup or self.evidently_metrics[name][0].decision_criterion.needs_calibration
            ]
        )
        report.run(
            reference_data=embeddings_ref,
            current_data=embeddings_cur,
            column_mapping=column_mapping,
        )
        results_raw = report.as_dict()

        metric_names = list(self.metrics_config)
        results = {
            metric_names[metric_idx]: MetricResult(
                metric_id=metric_result["metric"],
                # will be overwritten by DecisionPolicy inside the DataDriftTrigger
                is_drift=metric_result["result"]["drift_detected"],
                distance=metric_result["result"]["drift_score"],
            )
            for metric_idx, metric_result in enumerate(results_raw["metrics"])
        }
        
        # Compute the results for Evidently metrics that aren't supported through the EmbeddingsDriftMetric interface
        additional_metric_type_results = _evidently_addtional_metric_computation(
            self.evidently_metrics_config, embeddings_ref, embeddings_cur
        )
        
        return {**results, **additional_metric_type_results}


# -------------------------------------------------------------------------------------------------------------------- #
#                                                       Internal                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


def _get_evidently_metrics(
    metrics_config: dict[str, EvidentlyDriftMetric],
) -> dict[str, tuple[EvidentlyDriftMetric, EmbeddingsDriftMetric]]:
    """This function instantiates an Evidently metric given metric
    configuration. If we want to support multiple metrics in the future, we can
    loop through the configurations.

    Evidently metric configurations follow exactly the four DriftMethods defined in embedding_drift_methods:
    model, distance, mmd, ratio
    If metric_name not given, we use the default 'model' metric.
    Otherwise, we use the metric given by metric_name, with optional metric configuration specific to the metric.
    """
    metrics = {
        metric_ref: (
            config,
            EmbeddingsDriftMetric(EVIDENTLY_COLUMN_MAPPING_NAME, _evidently_metric_factory(config)),
        )
        for metric_ref, config in metrics_config.items()
        if config.id not in ["EvidentlyHellingerDistanceDriftMetric"]
    }
    return metrics


def _evidently_metric_factory(config: EvidentlyDriftMetric) -> EmbeddingsDriftMetric:
    if config.id == "EvidentlyModelDriftMetric":
        assert config.bootstrap is False, "Bootstrap is not supported in EvidentlyModelDriftMetric."
        return embedding_drift_methods.model(
            threshold=config.threshold,
            bootstrap=False,
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
        assert config.bootstrap is False, "Bootstrap is not supported in EvidentlySimpleDistanceDriftMetric."
        return embedding_drift_methods.distance(
            dist=config.distance_metric,
            threshold=config.threshold,
            bootstrap=False,
            pca_components=config.num_pca_component,
            quantile_probability=config.quantile_probability,
        )

    raise NotImplementedError(f"Metric {config.id} is not supported in EvidentlyDriftMetric.")

def _evidently_addtional_metric_computation(
    configs: dict[str, EvidentlyDriftMetric],
    embeddings_ref: pd.DataFrame,
    embeddings_cur: pd.DataFrame,
) -> dict[str, MetricResult]:
    metric_results: dict[str, MetricResult] = {}
    for metric_ref, config in configs.items():
        if config.id == "EvidentlyHellingerDistanceDriftMetric":
            column_distances = [
                # [0]: Hellinger distance, [1]: decision with dummy threshold (False)
                _hellinger_distance(embeddings_ref[c], embeddings_cur[c], ColumnType.Numerical, 0)[0]
                for c in embeddings_ref.columns
            ]
            aggregated_distance = np.mean(column_distances)
            metric_results[metric_ref] = MetricResult(
                metric_id=metric_ref,
                is_drift=False,  # dummy, will be overwritten by DecisionPolicy inside the DataDriftTrigger
                distance=aggregated_distance,
            )
    return metric_results

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from modyn.config.schema.pipeline import MetricResult
from modyn.config.schema.pipeline.trigger.drift.modyn import ModynDriftMetric, ModynMmdDriftMetric
from modyn.supervisor.internal.triggers.drift.metrics.mmd import mmd2_two_sample_test

from .drift_detector import DriftDetector


class ModynDriftDetector(DriftDetector):
    def __init__(self, metrics_config: dict[str, ModynDriftMetric]):
        modyn_metrics_config = {
            metric_ref: config for metric_ref, config in metrics_config.items() if config.id.startswith("Modyn")
        }
        super().__init__(modyn_metrics_config)

    def init_detector(self) -> None:
        pass

    def detect_drift(
        self,
        embeddings_ref: pd.DataFrame | np.ndarray | torch.Tensor,
        embeddings_cur: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> dict[str, MetricResult]:
        assert isinstance(embeddings_ref, np.ndarray)
        assert isinstance(embeddings_cur, np.ndarray)

        results: dict[str, MetricResult] = {}

        for metric_ref, config in self.metrics_config.items():
            if isinstance(config, ModynMmdDriftMetric):
                results[metric_ref] = mmd2_two_sample_test(
                    reference_emb=embeddings_ref,
                    current_emb=embeddings_ref,
                    num_bootstraps=config.num_bootstraps,
                    pca_components=config.num_pca_component,
                    quantile_probability=config.quantile_probability,
                    device=config.device,
                    num_workers=config.num_workers,
                )

            raise NotImplementedError(f"Metric {config.id} is not supported in ModynDetectDriftMetric.")

        return results

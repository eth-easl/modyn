from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from modyn.config.schema.pipeline.trigger.drift.alibi_detect import AlibiDetectDriftMetric
from modyn.config.schema.pipeline.trigger.drift.result import MetricResult
from typing_extensions import override

from .drift_detector import DriftDetector


class AlibiDriftDetector(DriftDetector):
    def __init__(self, metrics_config: dict[str, AlibiDetectDriftMetric]):
        super().__init__(metrics_config)

    @override
    def init_detector(self) -> None:
        pass

    @override
    def detect_drift(
        self,
        embeddings_ref: pd.DataFrame | np.ndarray | torch.Tensor,
        embeddings_cur: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> dict[str, MetricResult]:
        for metric_id, metric_config in self.metrics_config.items():
            assert metric_id
            assert metric_config
            raise NotImplementedError()
        assert self.metrics_config, "No metrics configured for drift detection"
        return {}

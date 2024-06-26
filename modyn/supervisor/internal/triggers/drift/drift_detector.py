from abc import ABC

import numpy as np
import pandas as pd
import torch
from modyn.config.schema.pipeline.trigger.drift.config import DriftMetric
from modyn.config.schema.pipeline.trigger.drift.result import MetricResult


class DriftDetector(ABC):
    # to do: multiple strategies to select reference data

    def __init__(self, metrics_config: dict[str, DriftMetric]):
        self.metrics_config = metrics_config

    def init_detector(self) -> None:
        pass

    def detect_drift(
        self,
        embeddings_ref: pd.DataFrame | np.ndarray | torch.Tensor,
        embeddings_cur: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> dict[str, MetricResult]:
        raise NotImplementedError()

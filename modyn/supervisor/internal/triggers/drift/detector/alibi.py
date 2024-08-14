from __future__ import annotations

import alibi_detect.utils.pytorch
import numpy as np
import pandas as pd
import torch
from alibi_detect.cd import (
    ChiSquareDrift,
    CVMDrift,
    FETDrift,
    KSDrift,
    LSDDDrift,
    MMDDrift,
)

from modyn.config.schema.pipeline import (
    AlibiDetectDriftMetric,
    AlibiDetectMmdDriftMetric,
    MetricResult,
)
from modyn.config.schema.pipeline.trigger.drift.alibi_detect import (
    AlibiDetectCVMDriftMetric,
    AlibiDetectKSDriftMetric,
)

from .drift import DriftDetector

_AlibiMetrics = MMDDrift | ChiSquareDrift | KSDrift | CVMDrift | FETDrift | LSDDDrift


class AlibiDriftDetector(DriftDetector):
    def __init__(self, metrics_config: dict[str, AlibiDetectDriftMetric]):
        alibi_metrics_config = {
            metric_ref: config for metric_ref, config in metrics_config.items() if config.id.startswith("AlibiDetect")
        }
        super().__init__(alibi_metrics_config)

    def init_detector(self) -> None:
        pass

    # to do: reuse reference data, and the configured metrics (incl. kernels)

    def detect_drift(
        self,
        embeddings_ref: pd.DataFrame | np.ndarray | torch.Tensor,
        embeddings_cur: pd.DataFrame | np.ndarray | torch.Tensor,
        is_warmup: bool,
    ) -> dict[str, MetricResult]:
        assert isinstance(embeddings_ref, (np.ndarray | torch.Tensor))
        assert isinstance(embeddings_cur, (np.ndarray | torch.Tensor))
        embeddings_ref = (
            embeddings_ref.detach().cpu().numpy() if isinstance(embeddings_ref, torch.Tensor) else embeddings_ref
        )
        embeddings_cur = (
            embeddings_cur.detach().cpu().numpy() if isinstance(embeddings_cur, torch.Tensor) else embeddings_cur
        )

        results: dict[str, MetricResult] = {}

        for metric_ref, config in self.metrics_config.items():
            if is_warmup and not config.decision_criterion.needs_calibration:
                continue

            metric = _alibi_detect_metric_factory(config, embeddings_ref)
            result = metric.predict(embeddings_cur, return_p_val=True, return_distance=True)  # type: ignore

            # some metrics return a list of distances (for every sample) instead of a single distance
            # we take the mean of the distances to get a scalar distance value
            _dist = (
                float(result["data"]["distance"].mean())
                if isinstance(result["data"]["distance"], np.ndarray)
                else result["data"]["distance"]
            )
            _p_val = (
                float(result["data"]["p_val"].mean())
                if isinstance(result["data"]["p_val"], np.ndarray)
                else result["data"]["p_val"]
            )

            results[metric_ref] = MetricResult(
                metric_id=metric_ref,
                # will be overwritten by DecisionPolicy inside the DataDriftTrigger
                is_drift=result["data"]["is_drift"],
                distance=_dist,
                p_val=_p_val,
                threshold=result["data"].get("threshold"),
            )

        return results


# -------------------------------------------------------------------------------------------------------------------- #
#                                                       Internal                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


def _alibi_detect_metric_factory(config: AlibiDetectDriftMetric, embeddings_ref: np.ndarray | list) -> _AlibiMetrics:  # type: ignore
    kernel = None
    if isinstance(config, AlibiDetectMmdDriftMetric):
        kernel = getattr(alibi_detect.utils.pytorch, config.kernel)

    if isinstance(config, AlibiDetectMmdDriftMetric):
        assert kernel is not None
        return MMDDrift(
            x_ref=embeddings_ref,
            backend="pytorch",
            p_val=config.p_val,
            kernel=kernel,
            n_permutations=config.num_permutations or 1,
            device=config.device,
            configure_kernel_from_x_ref=config.configure_kernel_from_x_ref,
            x_ref_preprocessed=config.x_ref_preprocessed,
        )

    if isinstance(config, AlibiDetectKSDriftMetric):
        return KSDrift(
            x_ref=embeddings_ref,
            p_val=config.p_val,
            correction=config.correction,
            x_ref_preprocessed=config.x_ref_preprocessed,
        )

    if isinstance(config, AlibiDetectCVMDriftMetric):
        return CVMDrift(
            x_ref=embeddings_ref,
            p_val=config.p_val,
            correction=config.correction,
            x_ref_preprocessed=config.x_ref_preprocessed,
        )

    raise NotImplementedError(f"Metric {config.id} is not supported in AlibiDetectDriftMetric.")

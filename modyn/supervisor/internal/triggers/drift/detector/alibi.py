from __future__ import annotations

import alibi_detect.utils.pytorch
import numpy as np
import pandas as pd
import torch
from alibi_detect.cd import (
    ChiSquareDrift,
    ClassifierDrift,
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
    AlibiDetectChiSquareDriftMetric,
    AlibiDetectClassifierDriftMetric,
    AlibiDetectCVMDriftMetric,
    AlibiDetectFETDriftMetric,
    AlibiDetectKSDriftMetric,
    AlibiDetectLSDDDriftMetric,
)
from modyn.supervisor.internal.triggers.drift.classifier_models import (
    alibi_classifier_models,
)
from modyn.supervisor.internal.triggers.drift.detector.drift import DriftDetector

_AlibiMetrics = MMDDrift | ClassifierDrift | ChiSquareDrift | CVMDrift | FETDrift | KSDrift | LSDDDrift | MMDDrift


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
        embeddings_ref: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor,
        embeddings_cur: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor,
        is_warmup: bool,
    ) -> dict[str, MetricResult]:
        if isinstance(embeddings_ref, pd.DataFrame):
            embeddings_ref = embeddings_ref.to_numpy()
        if isinstance(embeddings_ref, torch.Tensor):
            embeddings_ref = embeddings_ref.detach().cpu().numpy()
        if isinstance(embeddings_cur, pd.DataFrame):
            embeddings_cur = embeddings_cur.to_numpy()
        if isinstance(embeddings_cur, torch.Tensor):
            embeddings_cur = embeddings_cur.detach().cpu().numpy()

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
            _threshold = (
                float(result["data"]["threshold"].mean())
                if isinstance(result["data"]["threshold"], np.ndarray)
                else result["data"]["threshold"]
            )
            results[metric_ref] = MetricResult(
                metric_id=metric_ref,
                # will be overwritten by DecisionPolicy inside the DataDriftTrigger
                is_drift=result["data"]["is_drift"],
                distance=_dist,
                p_val=_p_val,
                threshold=_threshold,
            )

        return results


# -------------------------------------------------------------------------------------------------------------------- #
#                                                       Internal                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


def _alibi_detect_metric_factory(config: AlibiDetectDriftMetric, embeddings_ref: np.ndarray | list) -> _AlibiMetrics:  # type: ignore
    kernel = None
    if isinstance(config, AlibiDetectMmdDriftMetric):
        kernel = getattr(alibi_detect.utils.pytorch, config.kernel)

    kwargs = {}
    if config.preprocessor:
        kwargs.update({"preprocess_fn": config.preprocessor.gen_preprocess_fn(config.device)})

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
            **kwargs,
        )

    if isinstance(config, AlibiDetectClassifierDriftMetric):
        return ClassifierDrift(
            embeddings_ref,
            alibi_classifier_models[config.classifier_id],
            backend="pytorch",
            p_val=config.p_val,
            preds_type="logits",
            device=config.device,
            **kwargs,
        )

    if isinstance(config, AlibiDetectKSDriftMetric):
        return KSDrift(
            x_ref=embeddings_ref,
            p_val=config.p_val,
            correction=config.correction,
            x_ref_preprocessed=config.x_ref_preprocessed,
            **kwargs,
        )

    if isinstance(config, AlibiDetectCVMDriftMetric):
        return CVMDrift(
            x_ref=embeddings_ref,
            p_val=config.p_val,
            correction=config.correction,
            x_ref_preprocessed=config.x_ref_preprocessed,
            **kwargs,
        )

    if isinstance(config, AlibiDetectLSDDDriftMetric):
        return LSDDDrift(
            x_ref=embeddings_ref,
            backend="pytorch",
            n_permutations=config.num_permutations or 1,
            p_val=config.p_val,
            correction=config.correction,
            x_ref_preprocessed=config.x_ref_preprocessed,
            device=config.device,
            **kwargs,
        )

    if isinstance(config, AlibiDetectFETDriftMetric):
        return FETDrift(
            x_ref=embeddings_ref,
            p_val=config.p_val,
            correction=config.correction,
            x_ref_preprocessed=config.x_ref_preprocessed,
            n_features=config.n_features,
            **kwargs,
        )

    if isinstance(config, AlibiDetectChiSquareDriftMetric):
        return ChiSquareDrift(
            x_ref=embeddings_ref,
            p_val=config.p_val,
            correction=config.correction,
            x_ref_preprocessed=config.x_ref_preprocessed,
            n_features=config.n_features,
            **kwargs,
        )

    raise NotImplementedError(f"Metric {config.id} is not supported in AlibiDetectDriftMetric.")

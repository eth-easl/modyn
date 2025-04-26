from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from modyn.config.schema.pipeline.evaluation.metrics import MetricConfig
from modyn.evaluator.internal.metric_factory import MetricFactory
from modyn.evaluator.internal.metrics import (
    AbstractDecomposableMetric,
    AbstractHolisticMetric,
)
from modyn.evaluator.internal.metrics.abstract_evaluation_metric import (
    AbstractEvaluationMetric,
)


def setup_metrics(
    metric_configs: list[MetricConfig],
) -> dict[str, AbstractEvaluationMetric]:
    metrics: dict[str, AbstractEvaluationMetric] = {}
    # need to make sure that the metric names are unique as they are used for identification.
    for config in metric_configs:
        metric = MetricFactory.get_evaluation_metric(config)
        if metric.get_name() not in metrics:
            metrics[metric.get_name()] = metric
        else:
            raise ValueError(f"Duplicate metric name {metric.get_name()} found in the configuration")
    return metrics


@dataclass
class EvaluationResult:
    num_samples: int
    metric_results: dict[str, float]
    metrics_data: dict[str, AbstractEvaluationMetric]


# pylint: disable=too-many-locals, too-many-branches
def perform_evaluation(
    model: Any,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    metrics: dict[str, AbstractEvaluationMetric],
    label_transformer_function: Callable | None = None,
    amp: bool = False,
    generative: bool = False,
) -> EvaluationResult:
    device_type = "cuda" if "cuda" in device else "cpu"
    contains_holistic_metric = MetricFactory.prepare_metrics(metrics)

    y_true = []
    y_score = []
    model.eval()
    num_samples = 0
    with torch.inference_mode():
        for batch in dataloader:
            data: torch.Tensor | dict
            if isinstance(batch[1], torch.Tensor):
                data = batch[1].to(device)
            elif isinstance(batch[1], dict):
                data: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
                for name, tensor in batch[1].items():
                    data[name] = tensor.to(device)
            else:
                raise ValueError(f"data type {type(batch[1])} not supported")

            if label_transformer_function is None:
                target = batch[2].to(device)
            else:
                target = label_transformer_function(batch[2]).to(device)

            batch_size = target.shape[0]

            with torch.autocast(device_type, enabled=amp):
                if generative:
                    target = target[:, :, 0]
                    target[target == model.tokenizer.pad_token_id] = -100

                    output = model(data, target)

                else:
                    output = model(data)

                for metric in metrics.values():
                    if isinstance(metric, AbstractDecomposableMetric):
                        metric.evaluate_batch(target, output, batch_size)

                if contains_holistic_metric:
                    y_true.append(target.detach().cpu())
                    y_score.append(output.detach().cpu())

            num_samples += batch_size

    if len(y_true) > 0:
        assert contains_holistic_metric  # We only track y_true in case of holistic metrics
        y_true = torch.cat(y_true)
        y_score = torch.cat(y_score)

        for metric in metrics.values():
            if isinstance(metric, AbstractHolisticMetric):
                metric.evaluate_dataset(y_true, y_score, num_samples)

    metric_result: dict[str, float] = {
        metric_name: metric.get_evaluation_result() for metric_name, metric in metrics.items()
    }

    return EvaluationResult(num_samples=num_samples, metric_results=metric_result, metrics_data=metrics)

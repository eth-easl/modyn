import io
import logging
import multiprocessing as mp
import os
import pathlib
import traceback

import torch

from modyn.evaluator.internal.dataset.evaluation_dataset import EvaluationDataset
from modyn.evaluator.internal.metric_factory import MetricFactory
from modyn.evaluator.internal.metrics import (
    AbstractDecomposableMetric,
    AbstractEvaluationMetric,
    AbstractHolisticMetric,
)
from modyn.evaluator.internal.utils import EvaluationInfo
from modyn.utils import LABEL_TRANSFORMER_FUNC_NAME, deserialize_function


class PytorchEvaluator:
    # pylint: disable=too-many-branches

    def __init__(self, evaluation_info: EvaluationInfo, logger: logging.Logger, metric_result_queue: mp.Queue) -> None:
        self.logger = logger
        self._evaluation_id = evaluation_info.evaluation_id
        self._metric_result_queue = metric_result_queue
        self._model = evaluation_info.model_handler(
            evaluation_info.model_configuration_dict, evaluation_info.device, evaluation_info.amp
        )
        self._load_state(evaluation_info.model_path)

        self._eval_info = evaluation_info

        self._label_transformer_function = deserialize_function(
            evaluation_info.label_transformer, LABEL_TRANSFORMER_FUNC_NAME
        )

        self._device = evaluation_info.device
        self._device_type = "cuda" if "cuda" in self._device else "cpu"
        self._amp = evaluation_info.amp

        self._info("Initialized PyTorch evaluator.")

    @staticmethod
    def _prepare_dataloader(
        evaluation_info: EvaluationInfo, start_timestamp: int | None, end_timestamp: int | None
    ) -> torch.utils.data.DataLoader:
        dataset = EvaluationDataset(
            evaluation_info.dataset_id,
            evaluation_info.bytes_parser,
            evaluation_info.transform_list,
            evaluation_info.storage_address,
            evaluation_info.evaluation_id,
            evaluation_info.tokenizer,
            start_timestamp,
            end_timestamp,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=evaluation_info.batch_size,
            num_workers=evaluation_info.num_dataloaders,
            timeout=60 if evaluation_info.num_dataloaders > 0 else 0,
        )

        return dataloader

    @staticmethod
    def _setup_metrics(metric_configurations: list[str]) -> list[AbstractEvaluationMetric]:
        metrics = []
        # need to make sure that the metric names are unique as they are used for identification.
        metric_names = set()
        for configuration_json in metric_configurations:
            metric = MetricFactory.get_evaluation_metric(configuration_json)
            if metric.get_name() not in metric_names:
                metrics.append(metric)
                metric_names.add(metric.get_name())
        return metrics

    def _info(self, msg: str) -> None:
        self.logger.info(f"[Evaluation {self._evaluation_id}] {msg}")

    def _debug(self, msg: str) -> None:
        self.logger.debug(f"[Evaluation {self._evaluation_id}] {msg}")

    def _load_state(self, path: pathlib.Path) -> None:
        assert path.exists(), "Cannot load state from non-existing file"

        self._info(f"Loading model state from {path}")
        with open(path, "rb") as state_file:
            checkpoint = torch.load(io.BytesIO(state_file.read()), map_location=torch.device("cpu"))

        assert "model" in checkpoint
        self._model.model.load_state_dict(checkpoint["model"])

        # delete trained model from disk
        path.unlink()

    # pylint: disable-next=too-many-locals
    def _single_interval_evaluate(self, dataloader: torch.utils.data.DataLoader, interval_idx: int) -> None:
        self._info(f"Process {os.getpid()} starts evaluation.")
        metrics = self._setup_metrics(self._eval_info.raw_metrics)
        contains_holistic_metric = MetricFactory.prepare_metrics(metrics)
        y_true = []
        y_score = []

        self._model.model.eval()
        num_samples = 0
        with torch.inference_mode():
            for batch in dataloader:
                data: torch.Tensor | dict
                if isinstance(batch[1], torch.Tensor):
                    data = batch[1].to(self._device)
                elif isinstance(batch[1], dict):
                    data: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
                    for name, tensor in batch[1].items():
                        data[name] = tensor.to(self._device)
                else:
                    raise ValueError(f"data type {type(batch[1])} not supported")

                if self._label_transformer_function is None:
                    target = batch[2].to(self._device)
                else:
                    target = self._label_transformer_function(batch[2]).to(self._device)

                batch_size = target.shape[0]

                with torch.autocast(self._device_type, enabled=self._amp):
                    output = self._model.model(data)

                    for metric in metrics:
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

            for metric in metrics:
                if isinstance(metric, AbstractHolisticMetric):
                    metric.evaluate_dataset(y_true, y_score, num_samples)

        metric_result = []
        for metric in metrics:
            metric_result.append((metric.get_name(), metric.get_evaluation_result()))
        self._info(f"Finished evaluation of {interval_idx}. Putting items into queue...")
        self._metric_result_queue.put((interval_idx, metric_result), timeout=30)
        self._info(
            f"Finished evaluation of {interval_idx}: {num_samples} samples. "
            f"Queue size = {self._metric_result_queue.qsize()}"
        )

    def evaluate(self) -> None:
        for idx, interval_idx in enumerate(self._eval_info.not_failed_interval_ids):
            self._info(f"Evaluating interval {idx + 1}/{len(self._eval_info.not_failed_interval_ids)} ({interval_idx})")
            interval = self._eval_info.all_evaluation_intervals[interval_idx]
            dataloader = self._prepare_dataloader(self._eval_info, interval[0], interval[1])
            self._single_interval_evaluate(dataloader, interval_idx)
            self._info(f"interval {idx + 1}/{len(self._eval_info.not_failed_interval_ids)} done")

        self._info("All intervals done!")


def evaluate(
    evaluation_info: EvaluationInfo,
    log_path: pathlib.Path,
    exception_queue: mp.Queue,
    metric_result_queue: mp.Queue,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path)
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)

    try:
        evaluator = PytorchEvaluator(evaluation_info, logger, metric_result_queue)
        evaluator.evaluate()
        logger.info("Evaluator returned.")
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        exception_queue.put(exception_msg)

        if evaluation_info.model_path.exists():
            logger.error("Deleting downloaded model after exception")
            evaluation_info.model_path.unlink()

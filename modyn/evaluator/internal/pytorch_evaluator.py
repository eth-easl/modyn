import io
import logging
import multiprocessing as mp
import os
import pathlib
import traceback
from typing import Union, Optional

import torch
from modyn.evaluator.internal.dataset.evaluation_dataset import EvaluationDataset
from modyn.evaluator.internal.metric_factory import MetricFactory
from modyn.evaluator.internal.metrics import AbstractDecomposableMetric, AbstractHolisticMetric
from modyn.evaluator.internal.utils import EvaluationInfo
from modyn.utils import LABEL_TRANSFORMER_FUNC_NAME, deserialize_function


class PytorchEvaluator:
    # pylint: disable=too-many-branches

    def __init__(self, evaluation_info: EvaluationInfo, logger: logging.Logger) -> None:
        self.logger = logger
        self._evaluation_id = evaluation_info.evaluation_id

        self._model = evaluation_info.model_handler(
            evaluation_info.model_configuration_dict, evaluation_info.device, evaluation_info.amp
        )
        self._load_state(evaluation_info.model_path)

        self._eval_info = evaluation_info

        self._metrics = evaluation_info.metrics
        self._label_transformer_function = deserialize_function(
            evaluation_info.label_transformer, LABEL_TRANSFORMER_FUNC_NAME
        )

        self._device = evaluation_info.device
        self._device_type = "cuda" if "cuda" in self._device else "cpu"
        self._amp = evaluation_info.amp

        self._contains_holistic_metric = MetricFactory.prepare_metrics(self._metrics)

        self._info("Initialized PyTorch evaluator.")

    def _prepare_dataloader(
            self, evaluation_info: EvaluationInfo, start_timestamp: Optional[int], end_timestamp: Optional[int]
    ) -> torch.utils.data.DataLoader:
        self._debug("Creating EvaluationDataset.")
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
        self._debug("Creating DataLoader.")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=evaluation_info.batch_size,
            num_workers=evaluation_info.num_dataloaders,
            timeout=60 if evaluation_info.num_dataloaders > 0 else 0,
        )

        return dataloader

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

    def _single_interval_evaluate(self, dataloader: torch.utils.data.DataLoader) -> None:
        self._info(f"Process {os.getpid()} starts evaluation.")

        y_true = []
        y_score = []

        self._model.model.eval()
        num_samples = 0
        with torch.inference_mode():
            for batch in dataloader:
                data: Union[torch.Tensor, dict]
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

                    for metric in self._metrics:
                        if isinstance(metric, AbstractDecomposableMetric):
                            metric.evaluate_batch(target, output, batch_size)

                    if self._contains_holistic_metric:
                        y_true.append(target.detach().cpu())
                        y_score.append(output.detach().cpu())

                num_samples += batch_size

        if len(y_true) > 0:
            assert self._contains_holistic_metric  # We only track y_true in case of holistic metrics
            y_true = torch.cat(y_true)
            y_score = torch.cat(y_score)

            for metric in self._metrics:
                if isinstance(metric, AbstractHolisticMetric):
                    metric.evaluate_dataset(y_true, y_score, num_samples)

        self._info(f"Finished evaluation: {num_samples} samples")

    def evaluate(self, metric_result_queue: mp.Queue) -> None:
        for interval in self._eval_info.evaluation_intervals:
            for metric in self._metrics:
                metric.reset_state()
            dataloader = self._prepare_dataloader(self._eval_info, interval[0], interval[1])
            self._single_interval_evaluate(dataloader)
            metric_result = []

            # We do this since we might also have just non-holistic metrics, in which case len(y_true) always is 0
            for metric in self._metrics:
                metric_result.append((metric.get_name(), metric.get_evaluation_result()))
                metric_result_queue.put(metric_result)


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
        evaluator = PytorchEvaluator(evaluation_info, logger)
        evaluator.evaluate(metric_result_queue)
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        exception_queue.put(exception_msg)

        if evaluation_info.model_path.exists():
            logger.error("Deleting downloaded model after exception")
            evaluation_info.model_path.unlink()

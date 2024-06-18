import io
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import traceback
from typing import Union

import torch
from modyn.evaluator.internal.dataset.evaluation_dataset import EvaluationDataset
from modyn.evaluator.internal.metric_factory import MetricFactory
from modyn.evaluator.internal.metrics import AbstractDecomposableMetric, AbstractHolisticMetric
from modyn.evaluator.internal.utils import EvaluationInfo, EvaluatorMessages
from modyn.utils import LABEL_TRANSFORMER_FUNC_NAME, deserialize_function


class PytorchEvaluator:
    # pylint: disable=too-many-branches

    def __init__(
        self,
        evaluation_info: EvaluationInfo,
        status_query_queue: mp.Queue,
        status_response_queue: mp.Queue,
        metric_result_queue: mp.Queue,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self._evaluation_id = evaluation_info.evaluation_id

        self._model = evaluation_info.model_handler(
            evaluation_info.model_configuration_dict, evaluation_info.device, evaluation_info.amp
        )
        self._load_state(evaluation_info.model_path)

        # setup dataloaders
        self._info("Setting up data loaders.")
        self._dataloader = self._prepare_dataloader(evaluation_info)

        self._metrics = evaluation_info.metrics
        self._label_transformer_function = deserialize_function(
            evaluation_info.label_transformer, LABEL_TRANSFORMER_FUNC_NAME
        )

        self._device = evaluation_info.device
        self._device_type = "cuda" if "cuda" in self._device else "cpu"
        self._amp = evaluation_info.amp

        self._status_query_queue = status_query_queue
        self._status_response_queue = status_response_queue
        self._metric_result_queue = metric_result_queue

        self._num_samples = 0
        self._contains_holistic_metric = MetricFactory.prepare_metrics(self._metrics)

        self._info("Initialized PyTorch evaluator.")

    def _prepare_dataloader(self, evaluation_info: EvaluationInfo) -> torch.utils.data.DataLoader:
        self._debug("Creating EvaluationDataset.")
        dataset = EvaluationDataset(
            evaluation_info.dataset_id,
            evaluation_info.bytes_parser,
            evaluation_info.transform_list,
            evaluation_info.storage_address,
            evaluation_info.evaluation_id,
            evaluation_info.tokenizer,
            evaluation_info.start_timestamp,
            evaluation_info.end_timestamp,
        )
        self._debug("Creating DataLoader.")
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=evaluation_info.batch_size, num_workers=evaluation_info.num_dataloaders, timeout=60
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

    def send_status_to_server(self, batch_number: int) -> None:
        self._status_response_queue.put({"num_batches": batch_number, "num_samples": self._num_samples})

    def evaluate(self) -> None:
        self._info(f"Process {os.getpid()} starts evaluation.")

        y_true = []
        y_score = []

        self._model.model.eval()
        with torch.inference_mode():
            batch_number = -1
            for batch_number, batch in enumerate(self._dataloader):
                # As empty() is unreliable
                # we try to fetch an element within 10ms. If there is no
                # element within that timeframe returned, we continue.
                try:
                    req = self._status_query_queue.get(timeout=0.01)
                    if req == EvaluatorMessages.STATUS_QUERY_MESSAGE:
                        self.send_status_to_server(batch_number)
                    else:
                        raise ValueError("Unknown message in the status query queue")
                except queue.Empty:
                    pass

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

                self._num_samples += batch_size

        if len(y_true) > 0:
            assert self._contains_holistic_metric  # We only track y_true in case of holistic metrics
            y_true = torch.cat(y_true)
            y_score = torch.cat(y_score)

            for metric in self._metrics:
                if isinstance(metric, AbstractHolisticMetric):
                    metric.evaluate_dataset(y_true, y_score, self._num_samples)

        # We do this since we might also have just non-holistic metrics, in which case len(y_true) always is 0
        for metric in self._metrics:
            self._metric_result_queue.put((metric.get_name(), metric.get_evaluation_result()))

        self._info(f"Finished evaluation: {self._num_samples} samples, {batch_number + 1} batches.")


def evaluate(
    evaluation_info: EvaluationInfo,
    log_path: pathlib.Path,
    exception_queue: mp.Queue,
    status_query_queue: mp.Queue,
    status_response_queue: mp.Queue,
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
        evaluator = PytorchEvaluator(
            evaluation_info, status_query_queue, status_response_queue, metric_result_queue, logger
        )
        evaluator.evaluate()
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        exception_queue.put(exception_msg)

        if evaluation_info.model_path.exists():
            logger.error("Deleting downloaded model after exception")
            evaluation_info.model_path.unlink()

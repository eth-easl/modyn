import io
import logging
import multiprocessing as mp
import os
import pathlib
import traceback
from typing import Any

import torch

from modyn.evaluator.internal.core_evaluation import perform_evaluation, setup_metrics
from modyn.evaluator.internal.dataset.evaluation_dataset import EvaluationDataset
from modyn.evaluator.internal.pytorch_lighttuner import PytorchTuner
from modyn.evaluator.internal.utils import EvaluationInfo, TuningInfo
from modyn.models.modular_adapters.modular_adapters import apply_adapters
from modyn.utils import LABEL_TRANSFORMER_FUNC_NAME, deserialize_function


class PytorchEvaluator:
    # pylint: disable=too-many-branches

    def __init__(
        self,
        evaluation_info: EvaluationInfo,
        logger: logging.Logger,
        metric_result_queue: mp.Queue,
    ) -> None:
        self.logger = logger
        self._evaluation_id = evaluation_info.evaluation_id
        self._metric_result_queue = metric_result_queue
        self._model = evaluation_info.model_handler(
            evaluation_info.model_configuration_dict,
            evaluation_info.device,
            evaluation_info.amp,
        )
        self._load_state(evaluation_info.model_path, evaluation_info.model_wrappers, evaluation_info.model_wrapper_args)

        self._eval_info = evaluation_info

        self._label_transformer_function = deserialize_function(
            evaluation_info.label_transformer, LABEL_TRANSFORMER_FUNC_NAME
        )

        self._device = evaluation_info.device
        self._amp = evaluation_info.amp

        self._info("Initialized PyTorch evaluator.")

    @staticmethod
    def _prepare_dataloader(
        evaluation_info: EvaluationInfo, start_timestamp: int | None, end_timestamp: int | None
    ) -> torch.utils.data.DataLoader:
        dataset = EvaluationDataset(
            dataset_id=evaluation_info.dataset_id,
            bytes_parser=evaluation_info.bytes_parser,
            bytes_parser_target=evaluation_info.bytes_parser_target,
            serialized_transforms=evaluation_info.transform_list,
            storage_address=evaluation_info.storage_address,
            evaluation_id=evaluation_info.evaluation_id,
            include_labels=not evaluation_info.generative,
            serialized_target_transforms=evaluation_info.serialized_transforms_target,
            tokenizer=evaluation_info.tokenizer,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            sequence_length=evaluation_info.seq_length,
        )
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

    def _load_state(self, path: pathlib.Path, model_wrappers: Any, model_wrapper_args: Any) -> None:
        assert path.exists(), "Cannot load state from non-existing file"

        self._info(f"Loading model state from {path}")
        with open(path, "rb") as state_file:
            checkpoint = torch.load(io.BytesIO(state_file.read()), map_location=torch.device("cpu"))

        assert "model" in checkpoint
        print(model_wrappers)
        self._model.model = apply_adapters(self._model.model, model_wrappers, model_wrapper_args)
        incompat=self._model.model.load_state_dict(checkpoint["model"],strict=False,)
        print("missing:", incompat.missing_keys)
        print("unexpected:", incompat.unexpected_keys)

        # delete trained model from disk
        path.unlink()

    # pylint: disable-next=too-many-locals
    def _single_interval_evaluate(self, dataloader: torch.utils.data.DataLoader, interval_idx: int) -> None:
        self._info(f"Process {os.getpid()} starts evaluation.")
        metrics = setup_metrics(self._eval_info.raw_metrics)

        eval_result = perform_evaluation(
            self._model.model,
            dataloader,
            self._device,
            metrics,
            self._label_transformer_function,
            self._amp,
            self._eval_info.generative,
        )

        self._info(f"Finished evaluation of {interval_idx}. Putting items into queue...")
        self._metric_result_queue.put((interval_idx, list(eval_result.metric_results.items())), timeout=30)
        self._info(
            f"Finished evaluation of {interval_idx}: {eval_result.num_samples} samples. "
            f"Queue size = {self._metric_result_queue.qsize()}"
        )

    def _light_tune(self, tuning_info: TuningInfo) -> None:
        tuner = PytorchTuner(
            tuning_info=tuning_info,
            logger=self.logger,
            model=self._model.model,
            storage_address=self._eval_info.storage_address,
        )
        tuner.train()

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
    light_tuning_info: TuningInfo | None = None,  # Dictionary to pass tuning parameters
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

        # Perform light tuning before evaluation if enabled
        if evaluation_info.light_tuning:
            logger.info("Performing light tuning before evaluation.")
            light_tuning_info = evaluation_info.tuning_info
            # Ensure light_tuning_info is valid
            if not isinstance(light_tuning_info, TuningInfo):
                raise ValueError("light_tuning_info must be a dictionary with tuning parameters.")

            evaluator._light_tune(light_tuning_info)  # Pass tuning info

            logger.info("Light tuning completed.")

        evaluator.evaluate()  # Run evaluation after tuning
        logger.info("Evaluator returned.")
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        exception_queue.put(exception_msg)

        if evaluation_info.model_path.exists():
            logger.error("Deleting downloaded model after exception")
            evaluation_info.model_path.unlink()

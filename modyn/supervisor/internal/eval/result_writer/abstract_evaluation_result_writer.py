from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from typing import Any

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import SingleEvaluationData


class AbstractEvaluationResultWriter(ABC):
    """
    Abstract class used to write evaluation results to the evaluation directory
    """

    def __init__(self, pipeline_id: int, trigger_id: int, eval_directory: pathlib.Path):
        self.pipeline_id = pipeline_id
        self.trigger_id = trigger_id
        self.eval_directory = eval_directory

    @abstractmethod
    def add_evaluation_data(self, dataset_id: str, dataset_size: int, evaluation_data: SingleEvaluationData) -> None:
        """
        Called whenever a metric results are available for a particular dataset.

        Args:
            dataset_id: the involved dataset.
            dataset_size: the size (amount of samples) of the dataset.
            evaluation_data: contains the metric results.
        """
        raise NotImplementedError()

    @abstractmethod
    def store_results(self) -> Any:
        """
        Called in the end to store the results.
        """
        raise NotImplementedError()

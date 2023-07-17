import json
import pathlib
from abc import ABC, abstractmethod

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluationData
from torch.utils.tensorboard import SummaryWriter


class AbstractEvaluationResultWriter(ABC):
    """
    Abstract class used to write evaluation results to the evaluation directory
    """

    def __init__(self, pipeline_id: int, trigger_id: int, eval_directory: pathlib.Path):
        self.pipeline_id = pipeline_id
        self.trigger_id = trigger_id
        self.eval_directory = eval_directory

    @abstractmethod
    def add_evaluation_data(self, dataset_id: str, dataset_size: int, evaluation_data: list[EvaluationData]) -> None:
        """
        Called whenever a metric results are available for a particular dataset.

        Args:
            dataset_id: the involved dataset.
            dataset_size: the size (amount of samples) of the dataset.
            evaluation_data: contains the metric results.
        """
        raise NotImplementedError()

    @abstractmethod
    def store_results(self) -> None:
        """
        Called in the end to store the results.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """
        Defines the name used in the pipeline config for this result writer
        """
        raise NotImplementedError()


class JsonResultWriter(AbstractEvaluationResultWriter):
    def __init__(self, pipeline_id: int, trigger_id: int, eval_directory: pathlib.Path):
        super().__init__(pipeline_id, trigger_id, eval_directory)
        self.results: dict = {"datasets": []}

    def add_evaluation_data(self, dataset_id: str, dataset_size: int, evaluation_data: list[EvaluationData]) -> None:
        dataset_results: dict = {"dataset_size": dataset_size, "metrics": []}
        for metric in evaluation_data:
            dataset_results["metrics"].append({"name": metric.metric, "result": metric.result})
        self.results["datasets"].append({dataset_id: dataset_results})

    def store_results(self) -> None:
        file_name = f"{self.pipeline_id}_{self.trigger_id}.eval"
        with open(self.eval_directory / file_name, "w+", encoding="utf-8") as output_file:
            json.dump(self.results, output_file)

    @staticmethod
    def get_name() -> str:
        return "json"


class TensorboardResultWriter(AbstractEvaluationResultWriter):
    def __init__(self, pipeline_id: int, trigger_id: int, eval_directory: pathlib.Path):
        super().__init__(pipeline_id, trigger_id, eval_directory)
        self.writer = SummaryWriter(log_dir=str(eval_directory))

    def add_evaluation_data(self, dataset_id: str, dataset_size: int, evaluation_data: list[EvaluationData]) -> None:
        for metric in evaluation_data:
            self.writer.add_scalar(
                f"pipeline_{self.pipeline_id}/{dataset_id}/{metric.metric}", metric.result, self.trigger_id
            )

    def store_results(self) -> None:
        self.writer.flush()

    @staticmethod
    def get_name() -> str:
        return "tensorboard"

import pathlib

from torch.utils.tensorboard import SummaryWriter

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluationData
from modyn.supervisor.internal.eval.result_writer.abstract_evaluation_result_writer import (
    AbstractEvaluationResultWriter,
)


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

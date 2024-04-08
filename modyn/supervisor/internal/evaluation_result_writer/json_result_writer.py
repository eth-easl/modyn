import json
import pathlib
import os

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluationData
from modyn.supervisor.internal.evaluation_result_writer import AbstractEvaluationResultWriter


class JsonResultWriter(AbstractEvaluationResultWriter):
    def __init__(self, pipeline_id: int, trigger_id: int, eval_directory: pathlib.Path):
        super().__init__(pipeline_id, trigger_id, eval_directory)
        self.results: dict = {"datasets": []}
        self.pipeline_eval_dir = self.eval_directory / str(self.pipeline_id)
        os.makedirs(self.pipeline_eval_dir, exist_ok=True)

    def add_evaluation_data(self, dataset_id: str, dataset_size: int, evaluation_data: list[EvaluationData]) -> None:
        dataset_results: dict = {"dataset_size": dataset_size, "metrics": []}
        for metric in evaluation_data:
            dataset_results["metrics"].append({"name": metric.metric, "result": metric.result})
        self.results["datasets"].append({dataset_id: dataset_results})

    def store_results(self) -> None:
        file_name = f"{self.pipeline_id}_{self.trigger_id}.eval"

        with open(self.pipeline_eval_dir / file_name, "w+", encoding="utf-8") as output_file:
            json.dump(self.results, output_file)

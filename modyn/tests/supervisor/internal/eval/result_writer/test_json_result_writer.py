import json
import pathlib
import tempfile

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluationData
from modyn.supervisor.internal.eval.result_writer import DedicatedJsonResultWriter


def test_json_writer():
    with tempfile.TemporaryDirectory() as path:
        eval_dir = pathlib.Path(path)
        writer = DedicatedJsonResultWriter(10, 15, eval_dir)
        writer.add_evaluation_data("mnist", 1000, [EvaluationData(metric="Accuracy", result=0.5)])
        writer.store_results()

        file_path = eval_dir / f"{10}_{15}.eval"
        assert file_path.exists() and file_path.is_file()

        with open(file_path, "r", encoding="utf-8") as eval_file:
            evaluation_results = json.load(eval_file)
            assert evaluation_results == json.loads(
                """{
                "datasets": [
                    {
                        "mnist": {
                            "dataset_size": 1000,
                            "metrics": [
                                {
                                    "name": "Accuracy",
                                    "result": 0.5
                                }
                            ]
                        }
                    }
                ]
            }"""
            )

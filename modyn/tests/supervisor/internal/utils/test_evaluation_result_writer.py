import json
import os
import pathlib
import tempfile

# pylint: disable=no-name-in-module
from unittest.mock import patch

from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluationData
from modyn.supervisor.internal.utils.evaluation_result_writer import JsonResultWriter, TensorboardResultWriter


def test_names():
    assert TensorboardResultWriter.get_name() == "tensorboard"
    assert JsonResultWriter.get_name() == "json"


def test_json_writer():
    with tempfile.TemporaryDirectory() as path:
        eval_dir = pathlib.Path(path)
        writer = JsonResultWriter(10, 15, eval_dir)
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


def test_tensorboard_writer():
    with tempfile.TemporaryDirectory() as path:
        eval_dir = pathlib.Path(path)
        writer = TensorboardResultWriter(10, 15, eval_dir)
        writer.add_evaluation_data("mnist", 1000, [EvaluationData(metric="Accuracy", result=0.5)])
        writer.store_results()

        assert len(os.listdir(eval_dir)) == 1

    with tempfile.TemporaryDirectory() as path:
        eval_dir = pathlib.Path(path)
        result_writer = TensorboardResultWriter(10, 15, eval_dir)

        with patch.object(result_writer.writer, "add_scalar") as add_method:
            result_writer.add_evaluation_data("mnist", 1000, [EvaluationData(metric="Accuracy", result=0.5)])

            assert add_method.call_args[0][0] == "pipeline_10/mnist/Accuracy"
            assert add_method.call_args[0][1] == 0.5
            assert add_method.call_args[0][2] == 15

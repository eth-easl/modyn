import os
import pathlib
import tempfile
from unittest.mock import patch

# pylint: disable=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import EvaluationData
from modyn.supervisor.internal.eval.result_writer import TensorboardResultWriter


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

import os
import pathlib

import pytest
from modyn.evaluator import Evaluator
from modyn.utils import validate_yaml

modyn_config = (
    pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "examples" / "modyn_config.yaml"
)
example_pipeline_config = (
    pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "examples" / "example-pipeline.yaml"
)
benchmark_pipeline_config = (
    pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent / "benchmark" / "mnist" / "mnist.yaml"
)


def get_invalid_modyn_config() -> dict:
    return {"invalid": "not_valid"}


def test_evaluator_init():
    evaluator = Evaluator(modyn_config)
    assert evaluator.config == modyn_config


def test_validate_config():
    model_storage = Evaluator(modyn_config)
    assert model_storage._validate_config()[0]


def test_validate_pipeline_config():
    schema_path = (
        pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "config" / "schema" / "pipeline-schema.yaml"
    )
    assert validate_yaml(example_pipeline_config, schema_path)[0]
    assert validate_yaml(benchmark_pipeline_config, schema_path)[0]


def test_invalid_config():
    with pytest.raises(ValueError):
        Evaluator(get_invalid_modyn_config())

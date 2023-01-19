# pylint: disable=unused-argument, no-name-in-module
import sys
from unittest.mock import patch

import grpc
import modyn.utils
import pytest
import yaml
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import GetSamplesRequest  # noqa: E402, E501, E611
from modyn.backend.selector.internal.grpc.grpc_handler import GRPCHandler
from modyn.backend.selector.selector_entrypoint import main
from modyn.backend.selector.selector_server import SelectorServer


def noop_constructor_mock(self, config: dict):
    pass


@patch.object(GRPCHandler, "_init_metadata", return_value=None)
@patch.object(modyn.utils, "grpc_connection_established", return_value=True)
@patch.object(GRPCHandler, "register_training", return_value=0)
@patch.object(GRPCHandler, "get_info_for_training", return_value=(8, 1))
@patch.object(GRPCHandler, "get_samples_by_metadata_query")
def test_prepare_training_set(
    test_get_samples_by_metadata_query,
    test_get_info_for_training,
    test_register_training,
    test__grpc_connection_established,
    test__init_metadata,
):
    sample_cfg = {
        "selector": {
            "port": "50056",
        },
        "metadata_database": {
            "port": "50054",
            "hostname": "backend",
        },
    }

    with open("modyn/config/examples/example-pipeline.yaml", "r", encoding="utf-8") as pipeline_file:
        pipeline_cfg = yaml.safe_load(pipeline_file)

    selector_server = SelectorServer(pipeline_cfg, sample_cfg)
    servicer = selector_server.grpc_server

    assert selector_server.selector.register_training(training_set_size=8, num_workers=1) == 0

    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_classes = [1, 1, 1, 1, 2, 2, 3, 3]
    all_scores = [1] * 8
    all_seens = [False] * 8

    test_get_samples_by_metadata_query.side_effect = [
        (all_samples, all_scores, all_seens, all_classes, all_samples),
        ([], [], [], [], []),
    ]

    assert set(
        servicer.get_sample_keys_and_weight(
            GetSamplesRequest(training_id=0, training_set_number=0, worker_id=0), None
        ).training_samples_subset
    ) == set(["a", "b", "c", "d", "e", "f", "g", "h"])


class DummyServer:
    def __init__(self, arg):
        pass

    def add_insecure_port(self, arg=None):
        pass

    def start(self):
        pass

    def wait_for_termination(self):
        pass

    def add_generic_rpc_handlers(self, arg=None):
        pass


@patch.object(grpc, "server", return_value=DummyServer(None))
@patch.object(GRPCHandler, "_init_metadata", return_value=None)
@patch.object(modyn.utils, "grpc_connection_established", return_value=True)
@patch.object(GRPCHandler, "register_training", return_value=0)
@patch.object(GRPCHandler, "get_info_for_training", return_value=(8, 1))
def test_main(
    get_info_for_training,
    register_training,
    _grpc_connection_established,
    _init_metadata,
    wait_for_terination,
):
    # testargs = ["selector_entrypoint.py", "modyn/config/config.yaml"]
    testargs = [
        "selector_entrypoint.py",
        "modyn/config/examples/modyn_config.yaml",
        "modyn/config/examples/example-pipeline.yaml",
    ]
    with patch.object(sys, "argv", testargs):
        main()


def test_main_raise():
    testargs = [
        "selector_entrypoint.py",
        "modyn/config/examples/example-pipeline.yaml",
        "modyn/config/config.yaml",
        "extra",
    ]
    with patch.object(sys, "argv", testargs):
        with pytest.raises(SystemExit):
            main()

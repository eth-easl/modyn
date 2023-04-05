# pylint: disable=unused-argument,redefined-outer-name
from unittest.mock import patch

import grpc
import pathlib
import yaml
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.utils import (
    dynamic_module_import,
    grpc_connection_established,
    model_available,
    trigger_available,
    validate_timestr,
    validate_yaml,
    current_time_millis,
    convert_timestr_to_seconds,
    package_available_and_can_be_imported,
    flatten
)


@patch.object(GRPCHandler, "init_storage", lambda self: None)
def test_connection_established_times_out():
    assert not grpc_connection_established(grpc.insecure_channel("1.2.3.4:42"), 0.5)

@patch("grpc.channel_ready_future")
def test_connection_established_works_mocked(test_channel_ready_future):
    # Pretty dumb test, needs E2E test with running server.

    class MockFuture:
        def result(self, timeout):
            return True

    test_channel_ready_future.return_value = MockFuture()
    assert grpc_connection_established(grpc.insecure_channel("1.2.3.4:42"))

def test_dynamic_module_import():
    assert dynamic_module_import('modyn.utils') is not None

def test_model_available():
    assert model_available('ResNet18')
    assert not model_available('NonExistingModel')

def test_trigger_available():
    assert trigger_available('TimeTrigger')
    assert not trigger_available('NonExistingTrigger')

def test_validate_yaml():
    with open(pathlib.Path('modyn') / 'config' / 'examples' / 'modyn_config.yaml', 'r', encoding='utf-8') as concrete_file:
        file = yaml.safe_load(concrete_file)
    assert validate_yaml(file, pathlib.Path('modyn') / 'config' / 'schema' / 'modyn_config_schema.yaml')[0]
    with open(pathlib.Path('modyn') / 'config' / 'examples' / 'example-pipeline.yaml', 'r', encoding='utf-8') as concrete_file:
        file = yaml.safe_load(concrete_file)
    assert validate_yaml(file, pathlib.Path('modyn') / 'config' / 'schema' / 'pipeline-schema.yaml')[0]
    assert not validate_yaml({}, pathlib.Path('modyn') / 'config' / 'schema' / 'modyn_config_schema.yaml')[0]

@patch('time.time', lambda: 0.4)
def test_current_time_millis():
    assert current_time_millis() == 400

def test_validate_timestr():
    assert validate_timestr('10s')
    assert validate_timestr('10m')
    assert validate_timestr('10h')
    assert validate_timestr('10d')
    assert not validate_timestr('10')
    assert not validate_timestr('10x')
    assert not validate_timestr('10s10m')

def test_convert_timestr_to_seconds():
    assert convert_timestr_to_seconds('10s') == 10
    assert convert_timestr_to_seconds('10m') == 600
    assert convert_timestr_to_seconds('10h') == 36000
    assert convert_timestr_to_seconds('10d') == 864000

def test_package_available_and_can_be_imported():
    assert package_available_and_can_be_imported('modyn')
    assert not package_available_and_can_be_imported('testpackage')

def test_flatten():
    assert flatten([[1,2,3,4]]) == [1,2,3,4]
    assert flatten([[1,2],[3,4]]) == [1,2,3,4]
    assert flatten([[1,2],[3,4],[5,6]]) == [1,2,3,4,5,6]

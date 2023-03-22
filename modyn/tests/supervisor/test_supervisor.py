# pylint: disable=unused-argument,redefined-outer-name
import pathlib
import typing
from unittest.mock import MagicMock, call, patch

import pytest
from modyn.supervisor import Supervisor
from modyn.supervisor.internal.grpc_handler import GRPCHandler


def get_minimal_pipeline_config() -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "training": {
            "gpus": 1,
            "device": "cpu",
            "dataloader_workers": 1,
            "use_previous_model": True,
            "initial_model": "random",
            "initial_pass": {"activated": False},
            "learning_rate": 0.1,
            "batch_size": 42,
            "optimizers": [
                {"name": "default1", "algorithm": "SGD", "source": "PyTorch", "param_groups": [{"module": "model"}]},
            ],
            "optimization_criterion": {"name": "CrossEntropyLoss"},
            "checkpointing": {"activated": False},
            "selection_strategy": {"name": "NewDataStrategy", "maximum_keys_in_memory": 10},
        },
        "data": {"dataset_id": "test", "bytes_parser_function": "def bytes_parser_function(x):\n\treturn x"},
        "trigger": {"id": "DataAmountTrigger", "trigger_config": {"data_points_for_trigger": 1}},
    }


def get_minimal_system_config() -> dict:
    return {}


def noop_constructor_mock(self, pipeline_config: dict, modyn_config: dict, replay_at: typing.Optional[int]) -> None:
    pass


def sleep_mock(duration: int):
    raise KeyboardInterrupt


@patch.object(GRPCHandler, "init_selector", return_value=None)
@patch.object(GRPCHandler, "init_storage", return_value=None)
@patch.object(GRPCHandler, "init_trainer_server", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(GRPCHandler, "dataset_available", return_value=True)
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
def get_non_connecting_supervisor(
    test_trainer_server_available,
    test_dataset_available,
    test_connection_established,
    test_init_trainer_server,
    test_init_storage,
    test_init_selector,
) -> Supervisor:
    supervisor = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    return supervisor


def test_initialization() -> None:
    get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter


@patch.object(GRPCHandler, "init_selector", return_value=None)
@patch.object(GRPCHandler, "init_trainer_server", return_value=None)
@patch.object(GRPCHandler, "init_storage", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(GRPCHandler, "dataset_available", return_value=False)
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
def test_constructor_throws_on_invalid_system_config(
    test_trainer_server_available,
    test_dataset_available,
    test_connection_established,
    test_init_storage,
    test_init_trainer_server,
    test_init_selector,
) -> None:
    with pytest.raises(ValueError, match="Invalid system configuration"):
        Supervisor(get_minimal_pipeline_config(), {}, None)


@patch.object(GRPCHandler, "init_selector", return_value=None)
@patch.object(GRPCHandler, "init_trainer_server", return_value=None)
@patch.object(GRPCHandler, "init_storage", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(GRPCHandler, "dataset_available", return_value=True)
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
def test_constructor_throws_on_invalid_pipeline_config(
    test_trainer_server_available,
    test_dataset_available,
    test_connection_established,
    test_init_storage,
    test_init_trainer_server,
    test_init_selector,
) -> None:
    with pytest.raises(ValueError, match="Invalid pipeline configuration"):
        Supervisor({}, get_minimal_system_config(), None)


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_validate_pipeline_config_schema():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # Check that our minimal pipeline config gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config_schema()

    # Check that an empty pipeline config gets rejected
    sup.pipeline_config = {}
    assert not sup.validate_pipeline_config_schema()

    # Check that an unknown model gets accepted because it has the correct schema
    # Semantic validation is done in another method
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["model"]["id"] = "UnknownModel"
    assert sup.validate_pipeline_config_schema()


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test__validate_training_options():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # Check that our minimal pipeline config gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    assert sup._validate_training_options()

    # Check that training without GPUs gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["gpus"] = 0
    assert not sup._validate_training_options()

    # Check that training with an invalid batch size gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["batch_size"] = -1
    assert not sup._validate_training_options()

    # Check that training with an invalid strategy gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["selection_strategy"]["name"] = "UnknownStrategy"
    assert not sup._validate_training_options()

    # Check that training with an invalid initial model gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["initial_model"] = "UnknownInitialModel"
    assert not sup._validate_training_options()

    # Check that training with an invalid reference for initial pass gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["initial_pass"]["activated"] = True
    sup.pipeline_config["training"]["initial_pass"]["reference"] = "UnknownRef"
    assert not sup._validate_training_options()

    # Check that training with an invalid amount for initial pass gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["initial_pass"]["activated"] = True
    sup.pipeline_config["training"]["initial_pass"]["reference"] = "amount"
    sup.pipeline_config["training"]["initial_pass"]["amount"] = 2
    assert not sup._validate_training_options()

    # Check that training with an valid amount for initial pass gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["initial_pass"]["activated"] = True
    sup.pipeline_config["training"]["initial_pass"]["reference"] = "amount"
    sup.pipeline_config["training"]["initial_pass"]["amount"] = 0.5
    assert sup._validate_training_options()


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_validate_pipeline_config_content():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # Check that our minimal pipeline config gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config_content()

    # Check that an empty pipeline config throws an exception
    # because there is no model defined
    with pytest.raises(KeyError):
        sup.pipeline_config = {}
        assert not sup.validate_pipeline_config_content()

    # Check that an unknown model gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["model"]["id"] = "UnknownModel"
    assert not sup.validate_pipeline_config_content()

    # Check that an unknown trigger gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["trigger"]["id"] = "UnknownTrigger"
    assert not sup.validate_pipeline_config_content()

    # Check that training without GPUs gets rejected
    # (testing that _validate_training_options gets called)
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["gpus"] = 0
    assert not sup.validate_pipeline_config_content()


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_validate_pipeline_config():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # Check that our minimal pipeline config gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config()

    # Check that an empty pipeline config gets rejected
    sup.pipeline_config = {}
    assert not sup.validate_pipeline_config()

    # Check that an unknown model gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["model"]["id"] = "UnknownModel"
    assert not sup.validate_pipeline_config()


@patch.object(GRPCHandler, "dataset_available", lambda self, did: did == "existing")
def test_dataset_available():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["data"]["dataset_id"] = "existing"
    assert sup.dataset_available()

    sup.pipeline_config["data"]["dataset_id"] = "nonexisting"
    assert not sup.dataset_available()


@patch.object(GRPCHandler, "dataset_available", lambda self, did: did == "existing")
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
def test_validate_system(test_trainer_server_available):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["data"]["dataset_id"] = "existing"
    assert sup.validate_system()

    sup.pipeline_config["data"]["dataset_id"] = "nonexisting"
    assert not sup.validate_system()


def test_get_dataset_selector_batch_size_given():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    sup.pipeline_config = get_minimal_pipeline_config()
    sup.modyn_config = {
        "storage": {
            "datasets": [{"name": "test", "selector_batch_size": 2048}, {"name": "test1", "selector_batch_size": 128}]
        }
    }
    sup.get_dataset_selector_batch_size()
    assert sup._selector_batch_size == 2048


def test_get_dataset_selector_batch_size_not_given():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    sup.pipeline_config = get_minimal_pipeline_config()
    sup.modyn_config = {"storage": {"datasets": [{"name": "test"}]}}
    sup.get_dataset_selector_batch_size()
    assert sup._selector_batch_size == 128


def test_shutdown_trainer():
    # TODO(MaxiBoether): implement
    pass


@patch.object(GRPCHandler, "get_new_data_since", return_value=[[(10, 42, 0), (11, 43, 1)]])
@patch.object(Supervisor, "_handle_new_data", return_value=False, side_effect=KeyboardInterrupt)
def test_wait_for_new_data(test__handle_new_data: MagicMock, test_get_new_data_since: MagicMock):
    # This is a simple test and does not the inclusivity filtering!
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    sup.wait_for_new_data(21)
    test_get_new_data_since.assert_called_once_with("test", 21)
    test__handle_new_data.assert_called_once_with([(10, 42, 0), (11, 43, 1)])


@patch.object(GRPCHandler, "get_new_data_since", return_value=[[(10, 42, 0)], [(11, 43, 1)]])
@patch.object(Supervisor, "_handle_new_data", return_value=False, side_effect=[None, KeyboardInterrupt])
def test_wait_for_new_data_batched(test__handle_new_data: MagicMock, test_get_new_data_since: MagicMock):
    # This is a simple test and does not the inclusivity filtering!
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    sup.wait_for_new_data(21)
    test_get_new_data_since.assert_called_once_with("test", 21)

    expected_calls = [
        call([(10, 42, 0)]),
        call([(11, 43, 1)]),
    ]

    assert test__handle_new_data.call_args_list == expected_calls


def test_wait_for_new_data_filtering():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    mocked__handle_new_data_return_vals = [True, True, KeyboardInterrupt]
    mocked_get_new_data_since = [
        [[(10, 42, 0), (11, 43, 0), (12, 43, 1)]],
        [[(11, 43, 0), (12, 43, 1), (13, 43, 2), (14, 45, 3)]],
        [[]],
        ValueError,
    ]

    handle_mock: MagicMock
    with patch.object(sup, "_handle_new_data", side_effect=mocked__handle_new_data_return_vals) as handle_mock:
        get_new_data_mock: MagicMock
        with patch.object(sup.grpc, "get_new_data_since", side_effect=mocked_get_new_data_since) as get_new_data_mock:
            sup.wait_for_new_data(21)

            assert handle_mock.call_count == 3
            assert get_new_data_mock.call_count == 3

            expected_handle_mock_arg_list = [
                call([(10, 42, 0), (11, 43, 0), (12, 43, 1)]),
                call([(13, 43, 2), (14, 45, 3)]),
                call([]),
            ]
            assert handle_mock.call_args_list == expected_handle_mock_arg_list

            expected_get_new_data_arg_list = [call("test", 21), call("test", 43), call("test", 45)]
            assert get_new_data_mock.call_args_list == expected_get_new_data_arg_list


def test_wait_for_new_data_filtering_batched():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    mocked__handle_new_data_return_vals = [True, True, True, True, True, KeyboardInterrupt]
    mocked_get_new_data_since = [
        [[(10, 42, 0), (11, 43, 0)], [(12, 43, 1)]],
        [[(11, 43, 0)], [(12, 43, 1), (13, 43, 2)], [(14, 45, 3)]],
        [[]],
        ValueError,
    ]

    handle_mock: MagicMock
    with patch.object(sup, "_handle_new_data", side_effect=mocked__handle_new_data_return_vals) as handle_mock:
        get_new_data_mock: MagicMock
        with patch.object(sup.grpc, "get_new_data_since", side_effect=mocked_get_new_data_since) as get_new_data_mock:
            sup.wait_for_new_data(21)

            assert handle_mock.call_count == 6
            assert get_new_data_mock.call_count == 3

            expected_handle_mock_arg_list = [
                call([(10, 42, 0), (11, 43, 0)]),
                call([(12, 43, 1)]),
                call([]),
                call([(13, 43, 2)]),
                call([(14, 45, 3)]),
                call([]),
            ]
            assert handle_mock.call_args_list == expected_handle_mock_arg_list

            expected_get_new_data_arg_list = [call("test", 21), call("test", 43), call("test", 45)]
            assert get_new_data_mock.call_args_list == expected_get_new_data_arg_list


def test__handle_new_data_with_batch():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup._selector_batch_size = 3
    new_data = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (15, 6), (16, 7), (17, 8)]

    batch_mock: MagicMock
    with patch.object(sup, "_handle_new_data_batch") as batch_mock:
        sup._handle_new_data(new_data)
        expected_handle_new_data_batch_arg_list = [
            call([(10, 1), (11, 2), (12, 3)]),
            call([(13, 4), (14, 5), (15, 6)]),
            call([(16, 7), (17, 8)]),
        ]
        assert batch_mock.call_args_list == expected_handle_new_data_batch_arg_list


def test__handle_new_data_with_large_batch():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    new_data = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (15, 6), (16, 7), (17, 8)]

    batch_mock: MagicMock
    with patch.object(sup, "_handle_new_data_batch") as batch_mock:
        sup._handle_new_data(new_data)
        expected_handle_new_data_batch_arg_list = [call(new_data)]
        assert batch_mock.call_args_list == expected_handle_new_data_batch_arg_list


def test__handle_new_data():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    sup._selector_batch_size = 2
    batching_return_vals = [False, True, False]
    new_data = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5)]

    batch_mock: MagicMock
    with patch.object(sup, "_handle_new_data_batch", side_effect=batching_return_vals) as batch_mock:
        result = sup._handle_new_data(new_data)
        assert result

        expected_handle_new_data_batch_arg_list = [
            call([(10, 1), (11, 2)]),
            call([(12, 3), (13, 4)]),
            call([(14, 5)]),
        ]
        assert batch_mock.call_count == 3
        assert batch_mock.call_args_list == expected_handle_new_data_batch_arg_list


@patch.object(GRPCHandler, "inform_selector")
def test__handle_new_data_batch_no_triggers(test_inform_selector: MagicMock):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.pipeline_id = 42
    batch = [(10, 1), (11, 2)]

    with patch.object(sup.trigger, "inform", return_value=[]) as inform_mock:
        assert not sup._handle_new_data_batch(batch)

        inform_mock.assert_called_once_with(batch)
        test_inform_selector.assert_called_once_with(42, batch)


@patch.object(Supervisor, "_run_training")
@patch.object(GRPCHandler, "inform_selector_and_trigger")
@patch.object(GRPCHandler, "inform_selector")
def test__handle_triggers_within_batch(
    test_inform_selector: MagicMock, test_inform_selector_and_trigger: MagicMock, test__run_training: MagicMock
):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.pipeline_id = 42
    batch = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (15, 6), (16, 7)]
    triggering_indices = [1, 3, 5]
    trigger_ids = [0, 1, 2]
    test_inform_selector_and_trigger.side_effect = trigger_ids

    sup._handle_triggers_within_batch(batch, triggering_indices)

    inform_selector_and_trigger_expected_args = [
        call(42, [(10, 1), (11, 2)]),
        call(42, [(12, 3), (13, 4)]),
        call(42, [(14, 5), (15, 6)]),
    ]
    assert test_inform_selector_and_trigger.call_count == 3
    assert test_inform_selector_and_trigger.call_args_list == inform_selector_and_trigger_expected_args

    run_training_expected_args = [call(0), call(1), call(2)]
    assert test__run_training.call_count == 3
    assert test__run_training.call_args_list == run_training_expected_args

    assert test_inform_selector.call_count == 1
    test_inform_selector.assert_called_once_with(42, [(16, 7)])


@patch.object(Supervisor, "_run_training")
@patch.object(GRPCHandler, "inform_selector_and_trigger")
@patch.object(GRPCHandler, "inform_selector")
def test__handle_triggers_within_batch_empty_triggers(
    test_inform_selector: MagicMock, test_inform_selector_and_trigger: MagicMock, test__run_training: MagicMock
):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.pipeline_id = 42
    batch = [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5), (15, 6), (16, 7)]
    triggering_indices = [-1, -1, 3]
    trigger_ids = [0, 1, 2]
    test_inform_selector_and_trigger.side_effect = trigger_ids

    sup._handle_triggers_within_batch(batch, triggering_indices)

    inform_selector_and_trigger_expected_args = [
        call(42, []),
        call(42, []),
        call(42, [(10, 1), (11, 2), (12, 3), (13, 4)]),
    ]
    assert test_inform_selector_and_trigger.call_count == 3
    assert test_inform_selector_and_trigger.call_args_list == inform_selector_and_trigger_expected_args

    run_training_expected_args = [call(0), call(1), call(2)]
    assert test__run_training.call_count == 3
    assert test__run_training.call_args_list == run_training_expected_args

    assert test_inform_selector.call_count == 1
    test_inform_selector.assert_called_once_with(42, [(14, 5), (15, 6), (16, 7)])


@patch.object(GRPCHandler, "fetch_trained_model", return_value=pathlib.Path("/"))
@patch.object(GRPCHandler, "start_training", return_value=1337)
@patch.object(GRPCHandler, "wait_for_training_completion")
def test__run_training(
    test_wait_for_training_completion: MagicMock, test_start_training: MagicMock, test_fetch_trained_model: MagicMock
):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.pipeline_id = 42

    sup._run_training(21)

    assert sup.current_training_id == 1337

    test_wait_for_training_completion.assert_called_once_with(1337, 42, 21)
    test_start_training.assert_called_once_with(42, 21, get_minimal_pipeline_config(), None)
    test_fetch_trained_model.assert_called_once()


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_initial_pass():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # TODO(#10): implement a real test when func is implemented.
    sup.initial_pass()


@patch.object(GRPCHandler, "get_data_in_interval", return_value=[[(10, 1), (11, 2)]])
@patch.object(Supervisor, "_handle_new_data")
def test_replay_data_closed_interval(test__handle_new_data: MagicMock, test_get_data_in_interval: MagicMock):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.start_replay_at = 0
    sup.stop_replay_at = 42
    sup.replay_data()

    test_get_data_in_interval.assert_called_once_with("test", 0, 42)
    test__handle_new_data.assert_called_once_with([(10, 1), (11, 2)])


@patch.object(GRPCHandler, "get_data_in_interval", return_value=[[(10, 1)], [(11, 2)]])
@patch.object(Supervisor, "_handle_new_data")
def test_replay_data_closed_interval_batched(test__handle_new_data: MagicMock, test_get_data_in_interval: MagicMock):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.start_replay_at = 0
    sup.stop_replay_at = 42
    sup.replay_data()

    test_get_data_in_interval.assert_called_once_with("test", 0, 42)
    assert test__handle_new_data.call_count == 2
    assert test__handle_new_data.call_args_list == [call([(10, 1)]), call([(11, 2)])]


@patch.object(GRPCHandler, "get_new_data_since", return_value=[[(10, 1), (11, 2)]])
@patch.object(Supervisor, "_handle_new_data")
def test_replay_data_open_interval(test__handle_new_data: MagicMock, test_get_new_data_since: MagicMock):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.start_replay_at = 0
    sup.stop_replay_at = None
    sup.replay_data()

    test_get_new_data_since.assert_called_once_with("test", 0)
    test__handle_new_data.assert_called_once_with([(10, 1), (11, 2)])


@patch.object(GRPCHandler, "get_new_data_since", return_value=[[(10, 1)], [(11, 2)]])
@patch.object(Supervisor, "_handle_new_data")
def test_replay_data_open_interval_batched(test__handle_new_data: MagicMock, test_get_new_data_since: MagicMock):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.start_replay_at = 0
    sup.stop_replay_at = None
    sup.replay_data()

    test_get_new_data_since.assert_called_once_with("test", 0)
    assert test__handle_new_data.call_count == 2
    assert test__handle_new_data.call_args_list == [call([(10, 1)]), call([(11, 2)])]


@patch.object(GRPCHandler, "get_time_at_storage", return_value=21)
@patch.object(GRPCHandler, "register_pipeline_at_selector", return_value=42)
@patch.object(Supervisor, "get_dataset_selector_batch_size")
@patch.object(Supervisor, "initial_pass")
@patch.object(Supervisor, "replay_data")
@patch.object(Supervisor, "wait_for_new_data")
@patch.object(GRPCHandler, "unregister_pipeline_at_selector")
def test_non_experiment_pipeline(
    test_unregister_pipeline_at_selector: MagicMock,
    test_wait_for_new_data: MagicMock,
    test_replay_data: MagicMock,
    test_initial_pass: MagicMock,
    test_get_dataset_selector_batch_size: MagicMock,
    test_register_pipeline_at_selector: MagicMock,
    test_get_time_at_storage: MagicMock,
):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.experiment_mode = False
    sup.pipeline()

    test_get_time_at_storage.assert_called_once()
    test_register_pipeline_at_selector.assert_called_once()
    test_initial_pass.assert_called_once()
    test_get_dataset_selector_batch_size.assert_called_once()
    test_wait_for_new_data.assert_called_once_with(21)
    test_replay_data.assert_not_called()
    test_unregister_pipeline_at_selector.assert_called_once_with(42)


@patch.object(GRPCHandler, "get_time_at_storage", return_value=21)
@patch.object(GRPCHandler, "register_pipeline_at_selector", return_value=42)
@patch.object(Supervisor, "get_dataset_selector_batch_size")
@patch.object(Supervisor, "initial_pass")
@patch.object(Supervisor, "replay_data")
@patch.object(Supervisor, "wait_for_new_data")
@patch.object(GRPCHandler, "unregister_pipeline_at_selector")
def test_experiment_pipeline(
    test_unregister_pipeline_at_selector: MagicMock,
    test_wait_for_new_data: MagicMock,
    test_replay_data: MagicMock,
    test_initial_pass: MagicMock,
    test_get_dataset_selector_batch_size: MagicMock,
    test_register_pipeline_at_selector: MagicMock,
    test_get_time_at_storage: MagicMock,
):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.experiment_mode = True
    sup.pipeline()

    test_get_time_at_storage.assert_called_once()
    test_register_pipeline_at_selector.assert_called_once()
    test_initial_pass.assert_called_once()
    test_get_dataset_selector_batch_size.assert_called_once()
    test_wait_for_new_data.assert_not_called()
    test_replay_data.assert_called_once()
    test_unregister_pipeline_at_selector.assert_called_once_with(42)

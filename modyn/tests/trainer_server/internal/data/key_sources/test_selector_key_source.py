# pylint: disable=unused-argument, no-name-in-module
from unittest.mock import MagicMock, patch

import grpc
import pytest
from tenacity import RetryCallState

from modyn.selector.internal.grpc.generated.selector_pb2 import (
    GetSamplesRequest,
    NumberOfPartitionsResponse,
    SamplesResponse,
    UsesWeightsResponse,
)
from modyn.trainer_server.internal.dataset.key_sources import SelectorKeySource


def test_init():
    keysource = SelectorKeySource(12, 1, "localhost:1234")
    assert keysource._pipeline_id == 12
    assert keysource._trigger_id == 1
    assert keysource._selector_address == "localhost:1234"
    assert keysource._selectorstub is None
    assert keysource._uses_weights is None


class MockSelectorStub:
    def __init__(self, channel) -> None:
        pass

    def get_sample_keys_and_weights(self, request):
        request: GetSamplesRequest
        worker_id = request.worker_id
        partition_id = request.partition_id

        if worker_id == 0 and partition_id == 0:
            return [SamplesResponse(training_samples_subset=[1, 2, 3], training_samples_weights=[-1.0, -2.0, -3.0])]
        if worker_id == 0 and partition_id == 1:
            return [
                SamplesResponse(training_samples_subset=[10, 20, 30], training_samples_weights=[-10.0, -20.0, -30.0])
            ]
        if worker_id == 1 and partition_id == 0:
            return [
                SamplesResponse(
                    training_samples_subset=[100, 200, 300], training_samples_weights=[-100.0, -200.0, -300.0]
                )
            ]
        return [
            SamplesResponse(training_samples_subset=[110, 220, 330], training_samples_weights=[-110.0, -220.0, -330.0])
        ]

    def get_number_of_partitions(self, request):
        return NumberOfPartitionsResponse(num_partitions=2)

    def uses_weights(self, request):
        return UsesWeightsResponse(uses_weights=False)


class WeightedMockSelectorStub(MockSelectorStub):
    def uses_weights(self, request):
        return UsesWeightsResponse(uses_weights=True)


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", WeightedMockSelectorStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_weighted_key_source(test_grp, test_connection):
    keysource = SelectorKeySource(12, 1, "localhost:1234")
    assert keysource._selectorstub is None
    assert keysource._uses_weights is None
    with pytest.raises(AssertionError):
        keysource.uses_weights()
    with pytest.raises(AssertionError):
        keysource.get_num_data_partitions()

    keysource.init_worker()
    assert isinstance(keysource._selectorstub, WeightedMockSelectorStub)
    assert keysource._uses_weights
    assert keysource.uses_weights()
    assert keysource.get_num_data_partitions() == 2

    keys, weights = keysource.get_keys_and_weights(0, 0)
    assert keys == [1, 2, 3] and weights == [-1.0, -2.0, -3.0]
    keys, weights = keysource.get_keys_and_weights(0, 1)
    assert keys == [10, 20, 30] and weights == [-10.0, -20.0, -30.0]

    keys, weights = keysource.get_keys_and_weights(1, 0)
    assert keys == [100, 200, 300] and weights == [-100.0, -200.0, -300.0]
    keys, weights = keysource.get_keys_and_weights(1, 1)
    assert keys == [110, 220, 330] and weights == [-110.0, -220.0, -330.0]

    with pytest.raises(AssertionError):
        keysource._get_just_keys(None)

    # artificially switch to unweighted
    keysource._uses_weights = False
    keys, weights = keysource.get_keys_and_weights(0, 0)
    assert weights is None
    assert keys == [1, 2, 3]


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_unweighted_key_source(test_grp, test_connection):
    keysource = SelectorKeySource(12, 1, "localhost:1234")
    assert keysource._selectorstub is None
    assert keysource._uses_weights is None
    with pytest.raises(AssertionError):
        keysource.uses_weights()
    with pytest.raises(AssertionError):
        keysource.get_num_data_partitions()

    keysource.init_worker()
    assert isinstance(keysource._selectorstub, MockSelectorStub)
    assert not keysource._uses_weights
    assert not keysource.uses_weights()
    assert keysource.get_num_data_partitions() == 2

    keys, weights = keysource.get_keys_and_weights(0, 0)
    assert keys == [1, 2, 3] and weights is None
    keys, weights = keysource.get_keys_and_weights(0, 1)
    assert keys == [10, 20, 30] and weights is None

    keys, weights = keysource.get_keys_and_weights(1, 0)
    assert keys == [100, 200, 300] and weights is None
    keys, weights = keysource.get_keys_and_weights(1, 1)
    assert keys == [110, 220, 330] and weights is None

    with pytest.raises(AssertionError):
        keysource._get_both_keys_and_weights(None)

    # artificially switch to weighted
    keysource._uses_weights = True
    keys, weights = keysource.get_keys_and_weights(0, 0)
    assert weights == [-1.0, -2.0, -3.0]
    assert keys == [1, 2, 3]


def test_retry_reconnection_callback():
    pipeline_id = 12
    trigger_id = 1
    selector_address = "localhost:1234"
    keysource = SelectorKeySource(pipeline_id, trigger_id, selector_address)

    # Create a mock RetryCallState
    mock_retry_state = MagicMock(spec=RetryCallState)
    mock_retry_state.attempt_number = 3
    mock_retry_state.outcome = MagicMock()
    mock_retry_state.outcome.failed = True
    mock_retry_state.args = [keysource]

    # Mock the _connect_to_selector method to raise an exception
    with patch.object(
        keysource, "_connect_to_selector", side_effect=ConnectionError("Connection failed")
    ) as mock_method:
        # Call the retry_reconnection_callback with the mock state
        with pytest.raises(ConnectionError):
            SelectorKeySource.retry_reconnection_callback(mock_retry_state)

        # Check that the method tried to reconnect
        mock_method.assert_called()

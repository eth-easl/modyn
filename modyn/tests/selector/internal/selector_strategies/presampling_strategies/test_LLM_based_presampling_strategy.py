from unittest.mock import MagicMock, patch

import grpc

from modyn.config.schema.pipeline import PresamplingConfig

# Adjust this import according to your repository structure.
from modyn.selector.internal.selector_strategies.presampling_strategies import LLMEvaluationPresamplingStrategy


# --- Define a minimal FakeChannel ---
class FakeChannel:
    def subscribe(self, callback, try_to_connect=False):
        pass

    def unsubscribe(self, callback):
        pass

    def unary_stream(self, method, request_serializer, response_deserializer):
        return lambda req: []


class FakeStorageBackend:
    def __init__(self, raw_keys):
        self.raw_keys = raw_keys
        self.inserted_keys = []  # Filled when _execute_on_session runs an INSERT

    def _get_data_from_storage(self, selector_keys: list[int], dataset_id: str):
        response_time = 1
        yield (
            selector_keys,
            [f"sample text {k}".encode() for k in selector_keys],
            [f"target text {k}".encode() for k in selector_keys],
            [k * 10 for k in selector_keys],
            response_time,
        )


def get_config() -> PresamplingConfig:
    return PresamplingConfig(ratio=100, strategy="LLMEvaluation", dataset_id="dummy-dataset", api_key="dummy")


def get_modyn_config() -> dict:
    return {"dummy": "config"}


def create_strategy(storage_backend, batch_size=2, custom_prompt=""):
    config = get_config()
    strategy = LLMEvaluationPresamplingStrategy(
        presampling_config=config,
        modyn_config=get_modyn_config(),
        pipeline_id=10,
        storage_backend=storage_backend,
    )
    return strategy


# --- Tests for LLMEvaluationPresamplingStrategy ---


def test_constructor():
    storage_backend = FakeStorageBackend(raw_keys=[])
    strategy = create_strategy(storage_backend)
    # Verify that attributes are correctly set.
    assert strategy.batch_size == 10
    assert strategy.model_name == "meta-llama/Llama-3.3-70B-Instruct"
    assert strategy.dataset_id == "dummy-dataset"


def test_evaluate_batch_quality():
    storage_backend = FakeStorageBackend(raw_keys=[])
    strategy = create_strategy(storage_backend, batch_size=3)
    # Prepare a fake response from the chat completions endpoint.
    fake_response = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = "true\nfalse\ntrue"
    fake_response.choices = [fake_choice]
    with patch.object(strategy.client.chat.completions, "create", return_value=fake_response) as mock_create:
        result = strategy.evaluate_batch_quality([101, 102, 103], model_name="dummy-model", dataset_id="dummy-dataset")
        # Expect [True, False, True]
        assert result == [True, False, True]
        mock_create.assert_called_once()


# --- Test get_presampling_query using a FakeChannel ---
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=FakeChannel())
@patch.object(
    LLMEvaluationPresamplingStrategy,
    "evaluate_batch_quality",
    side_effect=[[False, True, True, False, True, False, False, False, True, False], [False, True]],
)
def test_get_presampling_query(mock_eval, mock_insecure_channel, mock_utils_grpc):
    storage_backend = FakeStorageBackend(raw_keys=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    strategy = create_strategy(storage_backend)
    query = strategy.get_presampling_query(next_trigger_id=5, tail_triggers=None, limit=None, trigger_dataset_size=100)
    compiled = str(query.compile(compile_kwargs={"literal_binds": True}))
    assert "temp_llm_filter_" in compiled
    assert "sample_key" in compiled
    assert storage_backend.inserted_keys == [2, 3, 5, 9, 12]

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

    # Define unary_stream if needed by StorageStub; for this test it is not actually called
    def unary_stream(self, method, request_serializer, response_deserializer):
        return lambda req: []


class FakeStorageBackend:
    """
    Mimics a storage backend that returns a 5-tuple:
    (keys, samples, targets, labels, response_time)
    and captures inserted keys when the presampling temporary table is populated.
    """

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

    def _execute_on_session(self, fn):
        import re

        class FakeBind:
            """Minimal engine/connection mock so create_all() doesn't crash."""

            def _run_ddl_visitor(self, visitor, element, **kwargs):
                pass

        class FakeScalars:
            def __init__(self, keys):
                self.keys = keys

            def all(self):
                return self.keys

        class FakeResult:
            def __init__(self, keys):
                self.keys = keys

            def scalars(self):
                return FakeScalars(self.keys)

        class FakeSession:
            """
            Manages a dict of table_name -> list of inserted sample_keys,
            so we can simulate CREATE TABLE, INSERT, and SELECT realistically.
            """

            def __init__(self, raw_keys, inserted_keys_ref):
                # For the real 'SelectorStateMetadata' selects, we return raw_keys.
                self.raw_keys = raw_keys

                # For the new “temp” tables, we track inserted rows in self.tables.
                self.tables = {}
                self.inserted_keys_ref = inserted_keys_ref

            def execute(self, statement, *multiparams, **params):
                # Convert statement to string for easy checking.
                statement_str = str(statement)
                statement_up = statement_str.upper()

                # Extract table names from the statement using quick regex searches.
                create_match = re.search(r"CREATE\s+TABLE\s+(\S+)", statement_str, re.IGNORECASE)
                insert_match = re.search(r"INSERT\s+INTO\s+(\S+)", statement_str, re.IGNORECASE)
                select_match = re.search(r"FROM\s+(\S+)", statement_str, re.IGNORECASE)

                if create_match:
                    table_name = create_match.group(1)
                    print(f"[FAKE] Creating table: {table_name}")
                    # Initialize an empty list of rows for that table.
                    self.tables[table_name] = []
                    return None

                if insert_match and "INSERT" in statement_up:
                    table_name = insert_match.group(1)
                    if multiparams and isinstance(multiparams[0], list):
                        # multiparams[0] = e.g. [{"sample_key": 2}, {"sample_key": 4}]
                        inserted = [d["sample_key"] for d in multiparams[0]]
                        # Append these keys to the table’s row list.
                        self.tables.setdefault(table_name, []).extend(inserted)
                        # Also store them in self.inserted_keys_ref for your test to check later.
                        self.inserted_keys_ref[:] = inserted
                        print(f"[FAKE] Inserting keys {inserted} into {table_name}")
                    return None

                if select_match and "SELECT" in statement_up:
                    table_name = select_match.group(1)
                    # Check if it’s a “temp” table or the real 'SelectorStateMetadata' table.
                    if table_name in self.tables:
                        # Return the keys inserted into that table.
                        result = self.tables[table_name]
                        print(f"[FAKE] Selecting from {table_name}, returning {result}")
                        return FakeResult(result)
                    # Possibly 'SelectorStateMetadata' or something else:
                    print(f"[FAKE] Selecting from {table_name}, returning raw_keys = {self.raw_keys}")
                    return FakeResult(self.raw_keys)

                return None

            def get_bind(self):
                return FakeBind()

            def commit(self):
                pass

        session_instance = FakeSession(self.raw_keys, self.inserted_keys)
        return fn(session_instance)


# --- Helper Functions ---
def get_config() -> PresamplingConfig:
    return PresamplingConfig(ratio=100, strategy="LLMEvaluation")


def get_modyn_config() -> dict:
    return {"dummy": "config"}


def create_strategy(storage_backend, batch_size=2, custom_prompt=""):
    config = get_config()
    strategy = LLMEvaluationPresamplingStrategy(
        presampling_config=config,
        modyn_config=get_modyn_config(),
        pipeline_id=10,
        storage_backend=storage_backend,
        batch_size=batch_size,
        model_name="dummy-model",
        ratio=100,
        custom_prompt=custom_prompt,
        api_key="dummy",
        base_url="http://dummy",
        dataset_id="dummy-dataset",
    )
    return strategy


# --- Tests for LLMEvaluationPresamplingStrategy ---


def test_constructor():
    storage_backend = FakeStorageBackend(raw_keys=[])
    strategy = create_strategy(storage_backend)
    # Verify that attributes are correctly set.
    assert strategy.batch_size == 2
    assert strategy.model_name == "dummy-model"
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
@patch.object(LLMEvaluationPresamplingStrategy, "evaluate_batch_quality", side_effect=[[False, True], [False, True]])
def test_get_presampling_query(mock_eval, mock_insecure_channel, mock_utils_grpc):
    storage_backend = FakeStorageBackend(raw_keys=[1, 2, 3, 4])
    strategy = create_strategy(storage_backend, batch_size=2)
    query = strategy.get_presampling_query(next_trigger_id=5, tail_triggers=None, limit=None, trigger_dataset_size=100)
    compiled = str(query.compile(compile_kwargs={"literal_binds": True}))
    assert "temp_llm_filter_" in compiled
    assert "sample_key" in compiled
    assert storage_backend.inserted_keys == [2, 4]

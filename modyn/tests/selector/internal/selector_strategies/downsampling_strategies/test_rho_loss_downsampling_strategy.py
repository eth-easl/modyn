import json
import os
import pathlib
import shutil
import tempfile
from typing import Any, Callable, List, Literal, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import pytest
from modyn.common.grpc.grpc_helpers import TrainerServerGRPCHandlerMixin
from modyn.config import SelectionStrategy as SelectionStrategyModel
from modyn.config.schema.pipeline import (
    DataConfig,
    ILTrainingConfig,
    OptimizationCriterion,
    OptimizerConfig,
    OptimizerParamGroup,
    RHOLossDownsamplingConfig,
)
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Pipeline, SelectorStateMetadata, TrainedModel, Trigger
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies.rho_loss_downsampling_strategy import (
    RHOLossDownsamplingStrategy,
)
from modyn.tests.selector.internal.storage_backend.utils import MockStorageBackend
from pydantic import TypeAdapter
from sqlalchemy import select

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"

TMP_DIR = tempfile.mkdtemp()


def get_minimal_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "hostname": "",
            "port": "0",
            "database": f"{database_path}",
        },
        "selector": {"insertion_threads": 8, "trigger_sample_directory": TMP_DIR},
    }


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield

    os.remove(database_path)
    shutil.rmtree(TMP_DIR)


@pytest.fixture
def il_training_config():
    return ILTrainingConfig(
        dataloader_workers=1,
        il_model_id="ResNet18",
        il_model_config={"num_classes": 2},
        use_previous_model=False,
        amp=False,
        device="cpu",
        batch_size=16,
        epochs_per_trigger=1,
        shuffle=True,
        optimizers=[
            OptimizerConfig(
                name="default",
                algorithm="SGD",
                source="PyTorch",
                param_groups=[OptimizerParamGroup(module="model", config={"lr": 0.01})],
            )
        ],
        optimization_criterion=OptimizationCriterion(name="CrossEntropyLoss"),
    )


@pytest.fixture
def data_config():
    return DataConfig(
        dataset_id="test",
        bytes_parser_function="def bytes_parser_function(x):\n\treturn x",
    )


def noop_init_trainer_server(self):
    return


def store_samples(
    pipeline_id: int, trigger_id: int, key_ts_label_tuples: List[Tuple[int, int, int]], tmp_version=0
) -> None:
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        for key, timestamp, label in key_ts_label_tuples:
            database.session.add(
                SelectorStateMetadata(
                    pipeline_id=pipeline_id,
                    sample_key=key,
                    timestamp=timestamp,
                    label=label,
                    seen_in_trigger_id=trigger_id,
                    tmp_version=tmp_version,
                )
            )
        database.session.commit()


def register_pipeline(auxiliary_pipeline_id: Optional[int], data_config: DataConfig) -> int:
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        pipeline_id = database.register_pipeline(
            num_workers=1,
            model_class_name="ResNet18",
            model_config=json.dumps({"num_classes": 2}),
            amp=False,
            selection_strategy="{}",
            data_config=data_config.model_dump_json(by_alias=True),
            full_model_strategy=ModelStorageStrategyConfig(name="PyTorchFullModel"),
            auxiliary_pipeline_id=auxiliary_pipeline_id,
        )
        database.session.commit()
    return pipeline_id


@patch.object(TrainerServerGRPCHandlerMixin, "init_trainer_server", noop_init_trainer_server)
@patch.object(AbstractSelectionStrategy, "store_training_set", return_value=(42, 42, {}))
def test__persist_holdout_set(
    mock_store_training_set,
    il_training_config: ILTrainingConfig,
    data_config: DataConfig,
):
    pipeline_id = register_pipeline(None, data_config)

    modyn_config = get_minimal_modyn_config()

    downsampling_config = RHOLossDownsamplingConfig(
        ratio=60,
        holdout_set_ratio=50,
        il_training_config=il_training_config,
        holdout_set_strategy="Simple",
    )
    maximum_keys_in_memory = 4
    trigger_id = 3
    dataset_range = (13, 37)
    store_samples(
        pipeline_id=pipeline_id,
        trigger_id=trigger_id,
        key_ts_label_tuples=[(i, i, 0) for i in range(*dataset_range)],
    )

    strategy = RHOLossDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
    rho_pipeline_id = strategy.rho_pipeline_id
    storage_backend = MockStorageBackend(pipeline_id, modyn_config, maximum_keys_in_memory)

    mock_store_training_set.reset_mock()
    test_query = (
        select(SelectorStateMetadata.sample_key)
        .filter(
            SelectorStateMetadata.pipeline_id == pipeline_id,
            SelectorStateMetadata.seen_in_trigger_id == trigger_id,
        )
        .limit(11)
    )
    strategy._persist_holdout_set(test_query, trigger_id, storage_backend)
    mock_store_training_set.assert_called_once_with(
        rho_pipeline_id,
        trigger_id,
        modyn_config,
        ANY,
        ANY,
    )
    producer = mock_store_training_set.call_args[0][3]

    chunks = list(producer())
    # verify the partition size
    for chunk_id, (chunk, _) in enumerate(chunks):
        # only the last chunk can have less than maximum_keys_in_memory number of samples
        if chunk_id == len(chunks) - 1:
            assert len(chunk) <= maximum_keys_in_memory
        else:
            assert len(chunk) == maximum_keys_in_memory
    # verify the number of partitions
    # expected value: ceil(floor(trigger_dataset_size / holdout_set_ratio) / maximum_keys_in_memory)
    assert len(chunks) == 3
    # verify the samples
    samples = [sample for (chunk, _) in chunks for sample in chunk]
    assert len(samples) == 11
    for sample in samples:
        # verify key
        assert dataset_range[0] <= sample[0] < dataset_range[1]
        # verify weight
        assert sample[1] == pytest.approx(1.0)


@patch.object(TrainerServerGRPCHandlerMixin, "init_trainer_server", noop_init_trainer_server)
def test__get_or_create_rho_pipeline_id_and_get_data_config_when_present(
    il_training_config: ILTrainingConfig, data_config: DataConfig
):
    # we create the main pipeline and rho pipeline and their link in db in advance
    rho_pipeline_id = register_pipeline(None, data_config)
    main_pipeline_id = register_pipeline(rho_pipeline_id, data_config)

    modyn_config = get_minimal_modyn_config()

    downsampling_config = RHOLossDownsamplingConfig(
        ratio=60,
        holdout_set_ratio=50,
        il_training_config=il_training_config,
        holdout_set_strategy="Simple",
    )

    with patch.object(RHOLossDownsamplingStrategy, "_create_rho_pipeline_id") as mock_create_rho_pipeline_id:
        strategy = RHOLossDownsamplingStrategy(downsampling_config, modyn_config, main_pipeline_id, 4)
        mock_create_rho_pipeline_id.assert_not_called()
        assert strategy.rho_pipeline_id == rho_pipeline_id
        assert strategy.data_config == data_config


@pytest.mark.parametrize("holdout_set_strategy", ["Simple", "Twin"])
@patch.object(TrainerServerGRPCHandlerMixin, "init_trainer_server", noop_init_trainer_server)
def test__get_or_create_rho_pipeline_id_when_absent(
    il_training_config: ILTrainingConfig, holdout_set_strategy: Literal["Simple", "Twin"], data_config: DataConfig
):
    pipeline_id = register_pipeline(None, data_config)

    modyn_config = get_minimal_modyn_config()
    downsampling_config = RHOLossDownsamplingConfig(
        ratio=60,
        holdout_set_ratio=50,
        il_training_config=il_training_config,
        holdout_set_strategy=holdout_set_strategy,
    )
    assert downsampling_config.holdout_set_strategy == holdout_set_strategy

    strategy = RHOLossDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, 4)
    assert strategy.data_config == data_config
    # check that the created rho pipeline id is stored in the database
    with MetadataDatabaseConnection(modyn_config) as database:
        main_pipeline = database.session.get(Pipeline, pipeline_id)
        assert main_pipeline.auxiliary_pipeline_id == strategy.rho_pipeline_id

        rho_pipeline = database.session.get(Pipeline, strategy.rho_pipeline_id)

        assert rho_pipeline.num_workers == il_training_config.dataloader_workers
        assert DataConfig.model_validate_json(rho_pipeline.data_config) == data_config
        assert rho_pipeline.amp == il_training_config.amp
        selection_strategy_config = TypeAdapter(SelectionStrategyModel).validate_json(rho_pipeline.selection_strategy)
        assert selection_strategy_config == strategy.il_model_dummy_selection_strategy
        assert rho_pipeline.full_model_strategy_name == strategy.IL_MODEL_STORAGE_STRATEGY.name
        assert rho_pipeline.full_model_strategy_zip == strategy.IL_MODEL_STORAGE_STRATEGY.zip
        assert rho_pipeline.full_model_strategy_zip_algorithm == strategy.IL_MODEL_STORAGE_STRATEGY.zip_algorithm
        assert rho_pipeline.full_model_strategy_config == strategy.IL_MODEL_STORAGE_STRATEGY.config

        if holdout_set_strategy == "Twin":
            assert rho_pipeline.model_class_name == "RHOLOSSTwinModel"
            assert json.loads(rho_pipeline.model_config) == {
                "rho_real_model_class": il_training_config.il_model_id,
                "rho_real_model_config": il_training_config.il_model_config,
            }
        else:
            assert rho_pipeline.model_class_name == il_training_config.il_model_id
            assert json.loads(rho_pipeline.model_config) == il_training_config.il_model_config


def add_trigger_and_model(pipeline_id: int, trigger_id: int):
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.session.add(Trigger(pipeline_id=pipeline_id, trigger_id=trigger_id))
        database.session.add(
            TrainedModel(
                pipeline_id=pipeline_id,
                trigger_id=trigger_id,
                model_path="",
                metadata_path="",
            )
        )
        database.session.commit()


@patch.object(TrainerServerGRPCHandlerMixin, "init_trainer_server", noop_init_trainer_server)
def test_downsampling_params(il_training_config: ILTrainingConfig, data_config: DataConfig):
    pipeline_id = register_pipeline(None, data_config)

    modyn_config = get_minimal_modyn_config()
    downsampling_config = RHOLossDownsamplingConfig(
        ratio=60,
        holdout_set_ratio=50,
        il_training_config=il_training_config,
        holdout_set_strategy="Simple",
    )
    maximum_keys_in_memory = 4

    strategy = RHOLossDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)

    for trigger_id in range(3):
        add_trigger_and_model(strategy.rho_pipeline_id, trigger_id)

    expected = {
        "downsampling_ratio": 60,
        "ratio_max": 100,
        "maximum_keys_in_memory": maximum_keys_in_memory,
        "sample_then_batch": False,
        "il_model_id": 3,
        "rho_pipeline_id": strategy.rho_pipeline_id,
    }
    assert strategy.downsampling_params == expected


@pytest.mark.parametrize("previous_model_id", [None, 21])
@patch.object(TrainerServerGRPCHandlerMixin, "start_training", return_value=42)
@patch.object(TrainerServerGRPCHandlerMixin, "wait_for_training_completion")
@patch.object(TrainerServerGRPCHandlerMixin, "store_trained_model", return_value=33)
@patch.object(TrainerServerGRPCHandlerMixin, "init_trainer_server", noop_init_trainer_server)
def test__train_il_model(
    mock_store_trained_model: MagicMock,
    mock_wait_for_training_completion: MagicMock,
    mock_start_training: MagicMock,
    il_training_config: ILTrainingConfig,
    data_config: DataConfig,
    previous_model_id: Optional[int],
):
    pipeline_id = register_pipeline(None, data_config)
    il_training_config.use_previous_model = previous_model_id is not None
    modyn_config = get_minimal_modyn_config()
    downsampling_config = RHOLossDownsamplingConfig(
        ratio=60,
        holdout_set_ratio=50,
        il_training_config=il_training_config,
        holdout_set_strategy="Simple",
    )
    maximum_keys_in_memory = 4

    strategy = RHOLossDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
    trigger_id = 1
    model_id = strategy._train_il_model(trigger_id, previous_model_id)
    mock_start_training.assert_called_once_with(
        pipeline_id=strategy.rho_pipeline_id,
        trigger_id=trigger_id,
        training_config=il_training_config,
        data_config=data_config,
        previous_model_id=previous_model_id,
    )
    mock_wait_for_training_completion.assert_called_once_with(mock_start_training.return_value)
    mock_store_trained_model.assert_called_once_with(mock_start_training.return_value)
    assert model_id == mock_store_trained_model.return_value


@patch(
    "modyn.selector.internal.selector_strategies.downsampling_strategies.rho_loss_downsampling_strategy.isinstance",
    return_value=True,
)
@patch.object(RHOLossDownsamplingStrategy, "_get_latest_il_model_id")
@patch.object(RHOLossDownsamplingStrategy, "_train_il_model", return_value=42)
@patch.object(RHOLossDownsamplingStrategy, "_get_sampling_query")
@patch.object(RHOLossDownsamplingStrategy, "_persist_holdout_set")
@patch.object(RHOLossDownsamplingStrategy, "_clean_tmp_version")
@patch.object(TrainerServerGRPCHandlerMixin, "init_trainer_server", noop_init_trainer_server)
@pytest.mark.parametrize("use_previous_model", [True, False])
def test_inform_next_trigger_simple(
    mock__clean_tmp_version: MagicMock,
    mock__persist_holdout_set: MagicMock,
    mock__get_sampling_query: MagicMock,
    mock__train_il_model: MagicMock,
    mock_get_latest_il_model_id: MagicMock,
    mock_is_instance: MagicMock,
    use_previous_model: bool,
    il_training_config: ILTrainingConfig,
    data_config: DataConfig,
):
    pipeline_id = register_pipeline(None, data_config)
    il_training_config.use_previous_model = use_previous_model
    if use_previous_model:
        expected_previous_model_id = 67
        mock_get_latest_il_model_id.return_value = expected_previous_model_id
    else:
        expected_previous_model_id = None
    modyn_config = get_minimal_modyn_config()
    downsampling_config = RHOLossDownsamplingConfig(
        ratio=60,
        holdout_set_ratio=50,
        il_training_config=il_training_config,
        holdout_set_strategy="Simple",
    )
    maximum_keys_in_memory = 4
    mock_query = MagicMock()
    mock__get_sampling_query.return_value = mock_query

    strategy = RHOLossDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
    next_trigger_id = 2
    storage_backend = MockStorageBackend(pipeline_id, modyn_config, maximum_keys_in_memory)
    strategy.inform_next_trigger(next_trigger_id, storage_backend)

    if use_previous_model:
        mock_get_latest_il_model_id.assert_called_once_with(strategy.rho_pipeline_id, modyn_config)
    else:
        mock_get_latest_il_model_id.assert_not_called()
    mock__train_il_model.assert_called_once_with(next_trigger_id, expected_previous_model_id)
    mock__clean_tmp_version.assert_called_once_with(pipeline_id, next_trigger_id, ANY)
    mock__get_sampling_query.assert_called_once_with(pipeline_id, next_trigger_id, pytest.approx(0.5), ANY)
    mock__persist_holdout_set.assert_called_once_with(mock_query, next_trigger_id, ANY)


def test__get_latest_il_model_id():
    modyn_config = get_minimal_modyn_config()
    rho_pipeline_id = 1
    assert RHOLossDownsamplingStrategy._get_latest_il_model_id(rho_pipeline_id, modyn_config) is None
    add_trigger_and_model(rho_pipeline_id, 0)
    assert RHOLossDownsamplingStrategy._get_latest_il_model_id(rho_pipeline_id, modyn_config) == 1
    add_trigger_and_model(rho_pipeline_id, 1)
    assert RHOLossDownsamplingStrategy._get_latest_il_model_id(rho_pipeline_id, modyn_config) == 2


@patch(
    "modyn.selector.internal.selector_strategies.downsampling_strategies.rho_loss_downsampling_strategy.isinstance",
    return_value=True,
)
def test__clean_tmp_version(mock_is_instance, data_config: DataConfig):
    modyn_config = get_minimal_modyn_config()

    def mock_storage_backend_execute_on_session_patch(session_callback: Callable) -> Any:
        with MetadataDatabaseConnection(modyn_config) as database:
            return session_callback(database.session)

    pipeline_id = register_pipeline(None, data_config)
    trigger_id = 2
    store_samples(pipeline_id, trigger_id, [(i, i, 0) for i in range(10, 20)], tmp_version=1)

    mock_storage_backend = MockStorageBackend(pipeline_id, modyn_config, 4)
    mock_storage_backend._execute_on_session = Mock(wraps=mock_storage_backend_execute_on_session_patch)
    RHOLossDownsamplingStrategy._clean_tmp_version(pipeline_id, trigger_id, mock_storage_backend)
    mock_storage_backend._execute_on_session.assert_called_once_with(ANY)
    with MetadataDatabaseConnection(modyn_config) as database:

        assert (
            database.session.query(SelectorStateMetadata)
            .filter(
                SelectorStateMetadata.tmp_version == 1,
                SelectorStateMetadata.pipeline_id == pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id == trigger_id,
            )
            .count()
        ) == 0

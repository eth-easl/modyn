import json
import os
import pathlib
import shutil
import tempfile
from typing import List, Tuple
from unittest.mock import ANY, patch

import pytest
from modyn.config import DataConfig, OptimizationCriterion, OptimizerConfig, OptimizerParamGroup
from modyn.config.schema.sampling.downsampling_config import ILTrainingConfig, RHOLossDownsamplingConfig
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.metadata_database.models.auxiliary_pipelines import AuxiliaryPipeline, Pipeline
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies.rho_loss_downsampling_strategy import (
    RHOLossDownsamplingStrategy,
)
from modyn.tests.selector.internal.storage_backend.utils import MockStorageBackend

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
        num_workers=1,
        il_model_id="ResNet18",
        il_model_config={"num_classes": 2},
        amp=False,
        device="cpu",
        batch_size=16,
        epochs=1,
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
def il_data_config():
    return DataConfig(
        dataset_id="test",
        bytes_parser_function="def bytes_parser_function(x):\n\treturn x",
    )


def store_samples(pipeline_id: int, trigger_id: int, key_ts_label_tuples: List[Tuple[int, int, int]]) -> None:
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        for key, timestamp, label in key_ts_label_tuples:
            database.session.add(
                SelectorStateMetadata(
                    pipeline_id=pipeline_id,
                    sample_key=key,
                    timestamp=timestamp,
                    label=label,
                    seen_in_trigger_id=trigger_id,
                )
            )
        database.session.commit()


@patch.object(AbstractSelectionStrategy, "store_training_set")
@patch(
    "modyn.selector.internal.selector_strategies.downsampling_strategies.rho_loss_downsampling_strategy"
    ".get_trigger_dataset_size"
)
def test__prepare_holdout_set(
    mock_get_trigger_dataset_size,
    mock_store_training_set,
    il_training_config: ILTrainingConfig,
    il_data_config: DataConfig,
):
    pipeline_id = 42
    rho_pipeline_id = 24
    modyn_config = get_minimal_modyn_config()

    downsampling_config = RHOLossDownsamplingConfig(
        ratio=60,
        holdout_set_ratio=50,
        il_training_config=il_training_config,
        il_data_config=il_data_config,
    )
    maximum_keys_in_memory = 4
    trigger_id2dataset_size = [13, 24, 5]

    trigger_id2range = [(0, 13), (13, 37), (37, 42)]
    store_samples(
        pipeline_id=pipeline_id,
        trigger_id=0,
        key_ts_label_tuples=[(i, i, 0) for i in range(*trigger_id2range[0])],
    )

    store_samples(
        pipeline_id=pipeline_id,
        trigger_id=1,
        key_ts_label_tuples=[(i, i, 0) for i in range(*trigger_id2range[1])],
    )

    store_samples(
        pipeline_id=pipeline_id,
        trigger_id=2,
        key_ts_label_tuples=[(i, i, 0) for i in range(*trigger_id2range[2])],
    )

    strategy = RHOLossDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
    storage_backend = MockStorageBackend(pipeline_id, modyn_config, maximum_keys_in_memory)

    def validate_training_set_producer(producer, trigger_id):
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
        expected_num_partitions = [2, 3, 1]
        assert len(chunks) == expected_num_partitions[trigger_id]
        # verify the samples
        samples = [sample for (chunk, _) in chunks for sample in chunk]
        expected_num_samples = [6, 12, 2]
        assert len(samples) == expected_num_samples[trigger_id], "Number of samples is not as expected."
        for sample in samples:
            # verify key
            assert trigger_id2range[trigger_id][0] <= sample[0] < trigger_id2range[trigger_id][1]
            # verify weight
            assert sample[1] == pytest.approx(1.0)

    for trigger_id in range(3):
        mock_get_trigger_dataset_size.reset_mock()
        mock_store_training_set.reset_mock()
        mock_get_trigger_dataset_size.return_value = trigger_id2dataset_size[trigger_id]

        strategy._prepare_holdout_set(trigger_id, rho_pipeline_id, storage_backend)
        mock_get_trigger_dataset_size.assert_called_once_with(storage_backend, pipeline_id, trigger_id, tail_triggers=0)
        mock_store_training_set.assert_called_once_with(
            rho_pipeline_id,
            trigger_id,
            modyn_config,
            ANY,
            ANY,
        )
        training_set_producer = mock_store_training_set.call_args[0][3]
        validate_training_set_producer(training_set_producer, trigger_id)


def register_pipeline() -> int:
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        pipeline_id = database.register_pipeline(
            num_workers=1,
            model_class_name="ResNet18",
            model_config=json.dumps({"num_classes": 2}),
            amp=False,
            selection_strategy="{}",
            full_model_strategy=ModelStorageStrategyConfig(name="PyTorchFullModel"),
        )
        database.session.commit()
    return pipeline_id


def test__get_or_create_rho_pipeline_id_when_present(
    il_training_config: ILTrainingConfig,
    il_data_config: DataConfig,
):
    # we create the main pipeline and rho pipeline and their link in db in advance
    pipeline_id = register_pipeline()
    rho_pipeline_id = register_pipeline()

    modyn_config = get_minimal_modyn_config()
    with MetadataDatabaseConnection(modyn_config) as database:
        database.session.add(AuxiliaryPipeline(pipeline_id=pipeline_id, auxiliary_pipeline_id=rho_pipeline_id))
        database.session.commit()

    downsampling_config = RHOLossDownsamplingConfig(
        ratio=60,
        holdout_set_ratio=50,
        il_training_config=il_training_config,
        il_data_config=il_data_config,
    )

    with patch.object(RHOLossDownsamplingStrategy, "_create_rho_pipeline_id") as mock_create_rho_pipeline_id:
        strategy = RHOLossDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, 4)
        mock_create_rho_pipeline_id.assert_not_called()
        assert strategy.rho_pipeline_id == rho_pipeline_id


def test__get_or_create_rho_pipeline_id_when_absent(
    il_training_config: ILTrainingConfig,
    il_data_config: DataConfig,
):
    pipeline_id = register_pipeline()

    modyn_config = get_minimal_modyn_config()
    downsampling_config = RHOLossDownsamplingConfig(
        ratio=60,
        holdout_set_ratio=50,
        il_training_config=il_training_config,
        il_data_config=il_data_config,
    )

    strategy = RHOLossDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, 4)

    # check that the created rho pipeline id is stored in the database
    with MetadataDatabaseConnection(modyn_config) as database:
        stored_rho_pipeline_id = (
            database.session.query(AuxiliaryPipeline.auxiliary_pipeline_id)
            .filter(AuxiliaryPipeline.pipeline_id == pipeline_id)
            .scalar()
        )
        assert stored_rho_pipeline_id == strategy.rho_pipeline_id

        rho_pipeline = database.session.query(Pipeline).filter(Pipeline.pipeline_id == strategy.rho_pipeline_id).first()

        assert rho_pipeline.num_workers == il_training_config.num_workers
        assert rho_pipeline.model_class_name == il_training_config.il_model_id
        assert json.loads(rho_pipeline.model_config) == il_training_config.il_model_config
        assert rho_pipeline.amp == il_training_config.amp
        assert rho_pipeline.selection_strategy == strategy.IL_MODEL_DUMMY_SELECTION_STRATEGY
        assert rho_pipeline.full_model_strategy_name == strategy.IL_MODEL_STORAGE_STRATEGY.name
        assert rho_pipeline.full_model_strategy_zip == strategy.IL_MODEL_STORAGE_STRATEGY.zip
        assert rho_pipeline.full_model_strategy_zip_algorithm == strategy.IL_MODEL_STORAGE_STRATEGY.zip_algorithm
        assert rho_pipeline.full_model_strategy_config == strategy.IL_MODEL_STORAGE_STRATEGY.config

import json
from typing import Any, Iterable, Optional, Tuple

from modyn.common.grpc.grpc_helpers import TrainerServerGRPCHandlerMixin
from modyn.config.schema.pipeline import DataConfig, NewDataStrategyConfig
from modyn.config.schema.sampling.downsampling_config import RHOLossDownsamplingConfig
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Pipeline, SelectorStateMetadata
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.selector.internal.selector_strategies.utils import get_trigger_dataset_size
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.selector.internal.storage_backend.database import DatabaseStorageBackend
from sqlalchemy import Select, func, select


class RHOLossDownsamplingStrategy(AbstractDownsamplingStrategy):

    IL_MODEL_STORAGE_STRATEGY = ModelStorageStrategyConfig(name="PyTorchFullModel")

    def __init__(
        self,
        downsampling_config: RHOLossDownsamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
    ):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        # The choice of the selection strategy does not matter as long as it does not downsample data
        # at the trainer server. (We don't want to further select on holdout set)
        self.IL_MODEL_DUMMY_SELECTION_STRATEGY = NewDataStrategyConfig(
            maximum_keys_in_memory=maximum_keys_in_memory,
            tail_triggers=0
        )
        self.holdout_set_ratio = downsampling_config.holdout_set_ratio
        self.il_training_config = downsampling_config.il_training_config
        self.grpc = TrainerServerGRPCHandlerMixin(modyn_config)
        self.grpc.init_trainer_server()
        self.remote_downsampling_strategy_name = "RemoteRHOLossDownsampling"
        rho_pipeline_id, data_config = self._get_or_create_rho_pipeline_id_and_get_data_config()
        self.rho_pipeline_id: int = rho_pipeline_id
        self.data_config = data_config
        self.il_model_id: Optional[int] = None

    def inform_next_trigger(self, next_trigger_id: int, selector_storage_backend: AbstractStorageBackend) -> None:
        if not isinstance(selector_storage_backend, DatabaseStorageBackend):
            raise ValueError("RHOLossDownsamplingStrategy requires a DatabaseStorageBackend")

        self._prepare_holdout_set(next_trigger_id, selector_storage_backend)
        self.il_model_id = self._train_il_model(next_trigger_id)

    @property
    def downsampling_params(self) -> dict:
        config = super().downsampling_params
        assert self.il_model_id is not None
        config["il_model_id"] = self.il_model_id
        return config

    def _train_il_model(self, trigger_id: int) -> int:
        training_id = self.grpc.start_training(
            pipeline_id=self.rho_pipeline_id,
            trigger_id=trigger_id,
            training_config=self.il_training_config,
            data_config=self.data_config,
            previous_model_id=None,
        )
        self.grpc.wait_for_training_completion(training_id)
        model_id = self.grpc.store_trained_model(training_id)
        return model_id

    def _get_or_create_rho_pipeline_id_and_get_data_config(self) -> Tuple[int, DataConfig]:

        with MetadataDatabaseConnection(self._modyn_config) as database:
            main_pipeline = database.session.get(Pipeline, self._pipeline_id)
            assert main_pipeline is not None
            data_config_str = main_pipeline.data_config
            if main_pipeline.auxiliary_pipeline_id is not None:
                rho_pipeline_id = main_pipeline.auxiliary_pipeline_id
            else:
                # register rho pipeline
                rho_pipeline_id = self._create_rho_pipeline_id(database, data_config_str)
                main_pipeline.auxiliary_pipeline_id = rho_pipeline_id
                database.session.commit()
        return rho_pipeline_id, DataConfig.model_validate_json(data_config_str)

    def _create_rho_pipeline_id(self, database: MetadataDatabaseConnection, data_config_str: str) -> int:
        # Actually we don't need to store configs in the database as we just need the existence of the rho pipline.
        # We fetch configs directly from the object fields.
        # But for consistency, it is no harm to store the correct configs instead of dummy value in the database.
        rho_pipeline_id = database.register_pipeline(
            num_workers=self.il_training_config.dataloader_workers,
            model_class_name=self.il_training_config.il_model_id,
            model_config=json.dumps(self.il_training_config.il_model_config),
            amp=self.il_training_config.amp,
            selection_strategy=self.IL_MODEL_DUMMY_SELECTION_STRATEGY.model_dump_json(by_alias=True),
            data_config=data_config_str,
            full_model_strategy=self.IL_MODEL_STORAGE_STRATEGY,
        )
        return rho_pipeline_id

    def _prepare_holdout_set(self, next_trigger_id: int, selector_storage_backend: AbstractStorageBackend) -> None:
        current_trigger_dataset_size = get_trigger_dataset_size(
            selector_storage_backend, self._pipeline_id, next_trigger_id, tail_triggers=0
        )

        holdout_set_size = max(int(current_trigger_dataset_size * self.holdout_set_ratio / 100), 1)

        stmt = self._get_holdout_sampling_query(self._pipeline_id, next_trigger_id, holdout_set_size).execution_options(
            yield_per=self.maximum_keys_in_memory
        )

        def training_set_producer() -> Iterable[tuple[list[tuple[int, float]], dict[str, Any]]]:
            with MetadataDatabaseConnection(self._modyn_config) as database:
                for chunk in database.session.execute(stmt).partitions():
                    samples = [res[0] for res in chunk]
                    yield [(sample, 1.0) for sample in samples], {}

        AbstractSelectionStrategy.store_training_set(
            self.rho_pipeline_id,
            next_trigger_id,
            self._modyn_config,
            training_set_producer,
            selector_storage_backend.insertion_threads,
        )

    @staticmethod
    def _get_holdout_sampling_query(main_pipeline_id: int, trigger_id: int, target_size: int) -> Select:
        return (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == main_pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id == trigger_id,
            )
            .order_by(func.random())  # pylint: disable=E1102
            .limit(target_size)
        )

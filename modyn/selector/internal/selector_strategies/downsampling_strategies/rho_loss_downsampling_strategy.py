import json
import logging
from typing import Any, Iterable, Optional, Tuple

from modyn.common.grpc.grpc_helpers import TrainerServerGRPCHandlerMixin
from modyn.config.schema.pipeline import DataConfig, NewDataStrategyConfig, RHOLossDownsamplingConfig
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Pipeline, SelectorStateMetadata, TrainedModel, Trigger
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.selector.internal.storage_backend.database import DatabaseStorageBackend
from sqlalchemy import Select, func, select, update
from sqlalchemy.orm.session import Session

logger = logging.getLogger(__name__)


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
        self.il_model_dummy_selection_strategy = NewDataStrategyConfig(
            maximum_keys_in_memory=maximum_keys_in_memory, tail_triggers=0
        )
        self.holdout_set_ratio = downsampling_config.holdout_set_ratio
        self.holdout_set_strategy = downsampling_config.holdout_set_strategy
        self.il_training_config = downsampling_config.il_training_config
        self.grpc = TrainerServerGRPCHandlerMixin(modyn_config)
        self.grpc.init_trainer_server()
        self.remote_downsampling_strategy_name = "RemoteRHOLossDownsampling"
        rho_pipeline_id, data_config = self._get_or_create_rho_pipeline_id_and_get_data_config()
        self.rho_pipeline_id: int = rho_pipeline_id
        self.data_config = data_config

    def inform_next_trigger(self, next_trigger_id: int, selector_storage_backend: AbstractStorageBackend) -> None:
        if not isinstance(selector_storage_backend, DatabaseStorageBackend):
            raise ValueError("RHOLossDownsamplingStrategy requires a DatabaseStorageBackend")

        probability = self.holdout_set_ratio / 100

        query = self._get_sampling_query(self._pipeline_id, next_trigger_id, probability, selector_storage_backend)

        self._persist_holdout_set(query, next_trigger_id, selector_storage_backend)

        if self.il_training_config.use_previous_model:
            if self.holdout_set_strategy == "Twin":
                raise NotImplementedError("Use previous model currently is not supported for Twin strategy")
            previous_model_id = self._get_latest_il_model_id(self.rho_pipeline_id, self._modyn_config)
        else:
            previous_model_id = None

        model_id = self._train_il_model(next_trigger_id, previous_model_id)
        if self.holdout_set_strategy == "Twin":
            second_query = self._get_rest_data_query(self._pipeline_id, next_trigger_id)
            self._persist_holdout_set(second_query, next_trigger_id, selector_storage_backend)
            self._train_il_model(next_trigger_id, model_id)
        self._clean_tmp_version(self._pipeline_id, next_trigger_id, selector_storage_backend)

    @staticmethod
    def _clean_tmp_version(
        main_pipeline_id: int, trigger_id: int, selector_storage_backend: AbstractStorageBackend
    ) -> None:
        assert isinstance(selector_storage_backend, DatabaseStorageBackend)

        def _session_callback(session: Session) -> None:
            session.query(SelectorStateMetadata).filter(
                SelectorStateMetadata.pipeline_id == main_pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id == trigger_id,
            ).update({"tmp_version": 0})
            session.commit()

        selector_storage_backend._execute_on_session(_session_callback)

    @staticmethod
    def _get_rest_data_query(main_pipeline_id: int, trigger_id: int) -> Select:
        stmt = select(SelectorStateMetadata.sample_key).filter(
            SelectorStateMetadata.pipeline_id == main_pipeline_id,
            SelectorStateMetadata.seen_in_trigger_id == trigger_id,
            SelectorStateMetadata.tmp_version == 0,
        )

        return stmt

    @property
    def downsampling_params(self) -> dict:
        config = super().downsampling_params
        config["rho_pipeline_id"] = self.rho_pipeline_id
        il_model_id = self._get_latest_il_model_id(self.rho_pipeline_id, self._modyn_config)
        assert il_model_id is not None
        config["il_model_id"] = il_model_id
        return config

    @staticmethod
    def _get_latest_il_model_id(rho_pipeline_id: int, modyn_config: dict) -> Optional[int]:
        with MetadataDatabaseConnection(modyn_config) as database:
            # find the maximal trigger id. This is the current trigger id.
            max_trigger_id = (
                database.session.query(func.max(Trigger.trigger_id))
                .filter(Trigger.pipeline_id == rho_pipeline_id)
                .scalar()
            )
            if max_trigger_id is None:
                return None

            # one pipeline id and one trigger id can only correspond to one model
            il_model_id = (
                database.session.query(TrainedModel.model_id)
                .filter(TrainedModel.pipeline_id == rho_pipeline_id, TrainedModel.trigger_id == max_trigger_id)
                .scalar()
            )
            assert il_model_id is not None
        return il_model_id

    def _train_il_model(self, trigger_id: int, previous_model_id: Optional[int]) -> int:
        training_id = self.grpc.start_training(
            pipeline_id=self.rho_pipeline_id,
            trigger_id=trigger_id,
            training_config=self.il_training_config,
            data_config=self.data_config,
            previous_model_id=previous_model_id,
        )
        self.grpc.wait_for_training_completion(training_id)
        model_id = self.grpc.store_trained_model(training_id)
        logger.info(f"Stored trained model {model_id} for trigger {trigger_id} in rho pipeline {self.rho_pipeline_id}")
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
        if self.holdout_set_strategy == "Twin":
            model_class_name = "RHOLOSSTwinModel"
            model_config = {
                "rho_real_model_class": self.il_training_config.il_model_id,
                "rho_real_model_config": self.il_training_config.il_model_config,
            }
        else:
            model_class_name = self.il_training_config.il_model_id
            model_config = self.il_training_config.il_model_config

        rho_pipeline_id = database.register_pipeline(
            num_workers=self.il_training_config.dataloader_workers,
            model_class_name=model_class_name,
            model_config=json.dumps(model_config),
            amp=self.il_training_config.amp,
            selection_strategy=self.il_model_dummy_selection_strategy.model_dump_json(by_alias=True),
            data_config=data_config_str,
            full_model_strategy=self.IL_MODEL_STORAGE_STRATEGY,
        )
        return rho_pipeline_id

    def _persist_holdout_set(
        self, query: Select, next_trigger_id: int, selector_storage_backend: AbstractStorageBackend
    ) -> None:
        stmt = query.execution_options(yield_per=self.maximum_keys_in_memory)

        def training_set_producer() -> Iterable[tuple[list[tuple[int, float]], dict[str, Any]]]:
            with MetadataDatabaseConnection(self._modyn_config) as database:
                for chunk in database.session.execute(stmt).partitions():
                    samples = [res[0] for res in chunk]
                    yield [(sample, 1.0) for sample in samples], {}

        total_keys_in_trigger, *_ = AbstractSelectionStrategy.store_training_set(
            self.rho_pipeline_id,
            next_trigger_id,
            self._modyn_config,
            training_set_producer,
            selector_storage_backend.insertion_threads,
        )
        logger.info(
            f"Stored {total_keys_in_trigger} keys in the holdout set for trigger {next_trigger_id} "
            f"in rho pipeline {self.rho_pipeline_id}"
        )

    @staticmethod
    def _get_sampling_query(
        main_pipeline_id: int, trigger_id: int, probability: float, selector_storage_backend: AbstractStorageBackend
    ) -> Select:
        assert isinstance(selector_storage_backend, DatabaseStorageBackend)

        def _session_callback(session: Session) -> None:
            session.execute(
                update(SelectorStateMetadata)
                .values(tmp_version=1)
                .where(SelectorStateMetadata.pipeline_id == main_pipeline_id)
                .where(SelectorStateMetadata.seen_in_trigger_id == trigger_id)
                .where(func.rand() < probability)
            )

        selector_storage_backend._execute_on_session(_session_callback)

        stmt = select(SelectorStateMetadata.sample_key).filter(
            SelectorStateMetadata.pipeline_id == main_pipeline_id,
            SelectorStateMetadata.seen_in_trigger_id == trigger_id,
            SelectorStateMetadata.tmp_version == 1,
        )

        return stmt

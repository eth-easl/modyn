import json
from typing import Any, Iterable

from modyn.config.schema.sampling.downsampling_config import RHOLossDownsamplingConfig
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.metadata_database.models.auxiliary_pipelines import AuxiliaryPipeline
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.selector.internal.selector_strategies.utils import get_trigger_dataset_size
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.selector.internal.storage_backend.database import DatabaseStorageBackend
from sqlalchemy import Select, func, select


class RHOLossDownsamplingStrategy(AbstractDownsamplingStrategy):

    IL_MODEL_STORAGE_STRATEGY = ModelStorageStrategyConfig(name="PyTorchFullModel")
    IL_MODEL_DUMMY_SELECTION_STRATEGY = "{}"

    def __init__(
        self,
        downsampling_config: RHOLossDownsamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
    ):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self.holdout_set_ratio = downsampling_config.holdout_set_ratio
        self.il_training_config = downsampling_config.il_training_config
        self.remote_downsampling_strategy_name = "RemoteRHOLossDownsampling"
        self.rho_pipeline_id: int = self._get_or_create_rho_pipeline_id()

    def inform_next_trigger(self, next_trigger_id: int, selector_storage_backend: AbstractStorageBackend) -> None:
        if not isinstance(selector_storage_backend, DatabaseStorageBackend):
            raise ValueError("RHOLossDownsamplingStrategy requires a DatabaseStorageBackend")

        self._prepare_holdout_set(next_trigger_id, self.rho_pipeline_id, selector_storage_backend)
        # Step 3: Issue training request to the trainer server, with pipeline_id as rho_pipeline_id and trigger_id
        # as next_trigger_id. Wait for the training to complete. Store the model. Record model id in
        # downsampling_params, so that it can be fetched and used for downsampling.
        raise NotImplementedError

    def _get_or_create_rho_pipeline_id(self) -> int:

        with MetadataDatabaseConnection(self._modyn_config) as database:
            aux_pipeline = database.session.get(AuxiliaryPipeline, self._pipeline_id)
            if aux_pipeline is not None:
                return aux_pipeline.auxiliary_pipeline_id

            # register rho pipeline
            rho_pipeline_id = self._create_rho_pipeline_id(database)
            database.session.commit()
        return rho_pipeline_id

    def _create_rho_pipeline_id(self, database: MetadataDatabaseConnection) -> int:
        # Actually we don't need to store configs in the database as we just need the existence of the rho pipline.
        # We fetch configs directly from the object fields.
        # But for consistency, it is no harm to store the correct configs instead of dummy value in the database.
        rho_pipeline_id = database.register_pipeline(
            num_workers=self.il_training_config.num_workers,
            model_class_name=self.il_training_config.il_model_id,
            model_config=json.dumps(self.il_training_config.il_model_config),
            amp=self.il_training_config.amp,
            selection_strategy=self.IL_MODEL_DUMMY_SELECTION_STRATEGY,
            full_model_strategy=self.IL_MODEL_STORAGE_STRATEGY,
        )
        database.session.add(AuxiliaryPipeline(pipeline_id=self._pipeline_id, auxiliary_pipeline_id=rho_pipeline_id))
        return rho_pipeline_id

    def _prepare_holdout_set(
        self, next_trigger_id: int, rho_pipeline_id: int, selector_storage_backend: AbstractStorageBackend
    ) -> None:
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
            rho_pipeline_id,
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

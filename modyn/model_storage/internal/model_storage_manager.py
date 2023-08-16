import json
import logging
import os
import pathlib
import tempfile
from typing import Optional

import torch
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Pipeline, TrainedModel
from modyn.model_storage.internal.utils import ModelStorageStrategy
from modyn.utils import current_time_millis, dynamic_module_import, unzip_file, zip_file

logger = logging.getLogger(__name__)


class ModelStorageManager:
    """
    Class used as manager of the model storage component. Implements all model storage related functionalities.
    """

    def __init__(self, modyn_config: dict, storage_dir: pathlib.Path):
        self._modyn_config = modyn_config
        self._storage_dir = storage_dir

    def store_model(self, pipeline_id: int, trigger_id: int, checkpoint_path: pathlib.Path) -> int:
        checkpoint = torch.load(checkpoint_path)

        model_storage_strategy = self.get_model_storage_strategy(pipeline_id)

        assert "model" in checkpoint
        state_dict = checkpoint["model"]
        local_model_filename = f"{current_time_millis()}_{pipeline_id}_{trigger_id}.model"
        model_path = self._storage_dir / local_model_filename
        parent_id = self._handle_new_model(pipeline_id, trigger_id, state_dict, model_path, model_storage_strategy)
        checkpoint.pop("model")

        # now checkpoint only contains optimizer state and metadata
        local_metadata_filename = f"{current_time_millis()}_{pipeline_id}_{trigger_id}.metadata.zip"
        metadata_path = self._storage_dir / local_metadata_filename

        with tempfile.NamedTemporaryFile() as temp_file:
            torch.save(checkpoint, temp_file)
            zip_file(pathlib.Path(temp_file.name), metadata_path)

        with MetadataDatabaseConnection(self._modyn_config) as database:
            return database.add_trained_model(
                pipeline_id, trigger_id, local_model_filename, local_metadata_filename, parent_id
            )

    def _handle_new_model(
        self,
        pipeline_id: int,
        trigger_id: int,
        state_dict: dict,
        model_path: pathlib.Path,
        model_storage_strategy: ModelStorageStrategy,
    ) -> Optional[int]:
        if model_storage_strategy.incremental_model_strategy and (
            model_storage_strategy.full_model_interval is None
            or trigger_id % model_storage_strategy.full_model_interval != 0
        ):
            prev_model: Optional[TrainedModel] = self._get_previous_model(pipeline_id, trigger_id)
            if prev_model:
                # handle incremental model storage
                previous_model_state = self._get_base_model_state(pipeline_id)

                # load previous model state
                self._reconstruct_model(prev_model.model_id, previous_model_state, model_storage_strategy)

                # store incremental model
                model_storage_strategy.incremental_model_strategy.save_model(
                    state_dict, previous_model_state, model_path
                )

                return prev_model.model_id
            logger.warning("Previous model is not available! Storing full model...")

        # handle full model storage
        model_storage_strategy.full_model_strategy.save_model(state_dict, model_path)
        return None

    def _get_base_model_state(self, pipeline_id: int) -> dict:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model_id, model_config, amp = database.get_model_configuration(pipeline_id)
        model_module = dynamic_module_import("modyn.models")
        assert hasattr(model_module, model_id), f"Model {model_id} not available."

        model_handler = getattr(model_module, model_id)
        return model_handler(json.loads(model_config), "cpu", amp).model.state_dict()

    def _reconstruct_model(
        self, model_id: int, model_state: dict, model_storage_strategy: ModelStorageStrategy
    ) -> None:
        # we recursively overwrite the model state
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model: TrainedModel = database.session.get(TrainedModel, model_id)
        if not model.parent_model:
            # base case: we can load a fully stored model
            model_storage_strategy.full_model_strategy.load_model(model_state, self._storage_dir / model.model_path)
            return

        # recursive step: we recurse to load the model state of the parent model
        self._reconstruct_model(model.parent_model, model_state, model_storage_strategy)

        # we apply the incremental strategy to load our model state
        model_storage_strategy.incremental_model_strategy.load_model(model_state, self._storage_dir / model.model_path)

    def _get_previous_model(self, pipeline_id: int, trigger_id: int) -> Optional[TrainedModel]:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            return (
                database.session.query(TrainedModel)
                .filter(TrainedModel.pipeline_id == pipeline_id, TrainedModel.trigger_id == trigger_id - 1)
                .first()
            )

    def load_model(self, model_id: int, metadata: bool) -> Optional[dict]:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model: Optional[TrainedModel] = database.session.get(TrainedModel, model_id)
            if model is None:
                logger.error(f"Model {model_id} does not exist.")
                return None
        model_storage_strategy = self.get_model_storage_strategy(model.pipeline_id)

        model_state = self._get_base_model_state(model.pipeline_id)
        self._reconstruct_model(model_id, model_state, model_storage_strategy)
        model_dict = {"model": model_state}

        if metadata:
            if not model.metadata_path:
                logger.error(f"Metadata not available for model {model_id}")
                return None
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file_path = pathlib.Path(temp_file.name)
                unzip_file(self._storage_dir / model.metadata_path, temp_file_path)
                metadata_dict = torch.load(temp_file_path)
            model_dict.update(metadata_dict)

        return model_dict

    def delete_model(self, model_id: int) -> bool:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model: Optional[TrainedModel] = database.session.get(TrainedModel, model_id)

            if model is None:
                logger.error(f"Trained model {model_id} was not found.")
                return False
            model_storage_strategy = self.get_model_storage_strategy(model.pipeline_id)
            child_state = self._get_base_model_state(model.pipeline_id)

            child: TrainedModel
            for child in model.children:
                assert child.pipeline_id == model.pipeline_id, "Pipeline does not match for parent and child model"

                self._reconstruct_model(child.model_id, child_state, model_storage_strategy)
                model_storage_strategy.full_model_strategy.save_model(child_state, self._storage_dir / child.model_path)
                database.session.query(TrainedModel).filter(TrainedModel.model_id == child.model_id).update(
                    {"parent_model": None}
                )

            os.remove(self._storage_dir / model.model_path)
            if model.metadata_path:
                os.remove(self._storage_dir / model.metadata_path)

            database.session.delete(model)
            database.session.commit()
        logger.info(f"Successfully deleted model {model_id} and converted child models to be fully stored.")
        return True

    def get_model_storage_strategy(self, pipeline_id: int) -> ModelStorageStrategy:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            pipeline: Pipeline = database.session.query(Pipeline).get(pipeline_id)

        strategy = ModelStorageStrategy(
            pipeline.full_model_strategy_name,
            pipeline.full_model_strategy_zip,
            pipeline.full_model_strategy_zip_algorithm,
            pipeline.full_model_strategy_config,
        )

        if pipeline.inc_model_strategy_name is not None:
            strategy.register_incremental_model_strategy(
                pipeline.inc_model_strategy_name,
                pipeline.inc_model_strategy_zip,
                pipeline.inc_model_strategy_zip_algorithm,
                pipeline.inc_model_strategy_config,
                pipeline.full_model_interval,
            )

        return strategy

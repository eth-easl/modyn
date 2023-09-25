import json
import logging
import os
import pathlib
import tempfile
from typing import Optional

import torch
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Pipeline, TrainedModel
from modyn.model_storage.internal.utils import ModelStoragePolicy
from modyn.utils import current_time_millis, dynamic_module_import, unzip_file, zip_file

logger = logging.getLogger(__name__)


class ModelStorageManager:
    """
    Class used as manager of the model storage component. Implements all model storage related functionalities.
    """

    def __init__(self, modyn_config: dict, storage_dir: pathlib.Path, ftp_dir: pathlib.Path):
        """
        Constructor of the model storage manager. It establishes a connection to the metadata database in order
        to store information related to the trained models.

        Args:
            modyn_config: the modyn configuration.
            storage_dir: path to the folder, in which the trained models are stored.
            ftp_dir: FTP directory, which is used as temporary folder for serving trained models.
        """
        self._modyn_config = modyn_config
        self._storage_dir = storage_dir
        self._ftp_dir = ftp_dir

    def store_model(self, pipeline_id: int, trigger_id: int, checkpoint_path: pathlib.Path) -> int:
        """
        Store the trained model contained in the checkpoint file to disk. It uses the model storage policy that is
        specified for the pipeline. Depending on the trigger id, it is either stored fully (according to full model
        strategy) or incrementally by using the incremental model strategy.

        Args:
            pipeline_id: the pipeline identifier for the model.
            trigger_id: the trigger associated with the model.
            checkpoint_path: path to the checkpoint containing the model.

        Returns:
            int: the model id which identifies the stored model.
        """
        checkpoint = torch.load(checkpoint_path)
        policy = self.get_model_storage_policy(pipeline_id)

        # split the model (stored under the "model" key) from metadata.
        assert "model" in checkpoint
        state_dict = checkpoint["model"]
        local_model_filename = f"{current_time_millis()}_{pipeline_id}_{trigger_id}.model"
        model_path = self._storage_dir / local_model_filename

        # handle the new model according to the model storage policy. If it is stored incrementally, we receive
        # the model id of the parent.
        parent_id = self._handle_new_model(pipeline_id, trigger_id, state_dict, model_path, policy)
        checkpoint.pop("model")

        # now checkpoint only contains optimizer state and metadata.
        local_metadata_filename = f"{current_time_millis()}_{pipeline_id}_{trigger_id}.metadata.zip"
        metadata_path = self._storage_dir / local_metadata_filename

        # zip the metadata file.
        with tempfile.NamedTemporaryFile(dir=self._ftp_dir) as temp_file:
            torch.save(checkpoint, temp_file)
            zip_file(pathlib.Path(temp_file.name), metadata_path)

        # add the new model to the database.
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
        policy: ModelStoragePolicy,
    ) -> Optional[int]:
        """
        Handle the new model according to the model storage policy.

        Args:
            pipeline_id: the pipeline, to which the model belongs.
            trigger_id: the trigger identifier associated with the model.
            state_dict: the model's state.
            model_path: path, under which the model must be stored.
            policy: the model storage policy applied to store the model.
        Returns:
            int: if the model is stored incrementally, the parent model id is returned.
        """

        # check whether we must apply the incremental storage strategy or the full model strategy.
        if policy.incremental_model_strategy and (
            policy.full_model_interval is None or trigger_id % policy.full_model_interval != 0
        ):
            parent_model_id: Optional[int] = self._get_parent_model_id(pipeline_id, trigger_id)
            if parent_model_id is not None:
                # store the model according to the incremental model strategy.
                parent_model_state = self._get_base_model_state(pipeline_id)

                # load model state of the parent model.
                self._reconstruct_model(parent_model_id, parent_model_state, policy)

                # finally store the model delta.
                policy.incremental_model_strategy.store_model(state_dict, parent_model_state, model_path)

                return parent_model_id
            logger.warning("Previous model is not available! Storing full model...")

        # store the model in its entirety.
        policy.full_model_strategy.store_model(state_dict, model_path)
        return None

    def _get_base_model_state(self, pipeline_id: int) -> dict:
        """
        Get the base model state associated with a pipeline.

        Args:
            pipeline_id: the involved pipeline.

        Returns:
            dict: the plain model state derived from the model architecture of the pipeline's models.
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model_class_name, model_config, amp = database.get_model_configuration(pipeline_id)
        model_module = dynamic_module_import("modyn.models")
        assert hasattr(model_module, model_class_name), f"Model {model_class_name} not available."

        model_handler = getattr(model_module, model_class_name)
        return model_handler(json.loads(model_config), "cpu", amp).model.state_dict()

    def _reconstruct_model(self, model_id: int, model_state: dict, policy: ModelStoragePolicy) -> None:
        """
        Reconstruct the model given the model state and the model storage policy.
        The function recursively call itself, if the model is stored as a model delta.
        In this case it first loads the (fully stored) parent model into the model state before overwriting it
        according to the incremental model storage policy.

        Args:
            model_id: the identifier of the model to be reconstructed.
            model_state: the plain model state (or the loaded parent model state).
            policy: the model storage policy containing the strategies.
        Returns:
            None: the model state is overwritten in order to minimize memory overhead.
        """

        # we recursively overwrite the model state.
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model: TrainedModel = database.session.get(TrainedModel, model_id)
        if not model.parent_model:
            # base case: we can load a fully stored model.
            policy.full_model_strategy.load_model(model_state, self._storage_dir / model.model_path)
            return

        # recursive step: we recurse to load the model state of the parent model.
        self._reconstruct_model(model.parent_model, model_state, policy)

        # we apply the incremental strategy to load our model state.
        policy.incremental_model_strategy.load_model(model_state, self._storage_dir / model.model_path)

    def _get_parent_model_id(self, pipeline_id: int, trigger_id: int) -> Optional[int]:
        """
        Get the id of the parent model given the trigger id of a pipeline.

        Args:
            pipeline_id: the pipeline that generated the model.
            trigger_id: the trigger associated with the model.

        Returns:
            Optional[int]: the parent model id (if it exists).
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            previous_model: TrainedModel = (
                database.session.query(TrainedModel)
                .filter(TrainedModel.pipeline_id == pipeline_id, TrainedModel.trigger_id == trigger_id - 1)
                .first()
            )

        if not previous_model:
            return None
        if previous_model.parent_model is None:
            return previous_model.model_id
        return previous_model.parent_model

    def load_model(self, model_id: int, metadata: bool) -> Optional[dict]:
        """
        Loads a given model and optionally, also appends the metadata.

        Args:
            model_id: the model identifier of the model.
            metadata: whether metadata should be loaded alongside.

        Returns:
            Optional[dict]: dictionary containing the model state and metadata if the model exists.
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model: Optional[TrainedModel] = database.session.get(TrainedModel, model_id)
            if model is None:
                logger.error(f"Model {model_id} does not exist.")
                return None
        policy = self.get_model_storage_policy(model.pipeline_id)

        # retrieve the model by loading its state dictionary.
        model_state = self._get_base_model_state(model.pipeline_id)
        self._reconstruct_model(model_id, model_state, policy)
        model_dict = {"model": model_state}

        # append the metadata to the dictionary if specified.
        if metadata:
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file_path = pathlib.Path(temp_file.name)
                unzip_file(self._storage_dir / model.metadata_path, temp_file_path)
                metadata_dict = torch.load(temp_file_path)
            model_dict.update(metadata_dict)

        return model_dict

    def delete_model(self, model_id: int) -> bool:
        """
        Deletes a given model id. Only works, if all depending models (children) are deleted.

        Args:
            model_id: the identifier of the model.

        Returns:
            bool: True, whenever deletion was successful.
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model: Optional[TrainedModel] = database.session.get(TrainedModel, model_id)

            if model is None:
                logger.error(f"Trained model {model_id} was not found.")
                return False

            children = model.children
            if len(children) > 0:
                child_ids = [str(child.model_id) for child in children]
                logger.info(f"Model {model_id} has depending child models: {', '.join(child_ids)}")
                return False

            os.remove(self._storage_dir / model.model_path)
            os.remove(self._storage_dir / model.metadata_path)

            database.session.delete(model)
            database.session.commit()
        logger.info(f"Successfully deleted model {model_id}.")
        return True

    def get_model_storage_policy(self, pipeline_id: int) -> ModelStoragePolicy:
        """
        Returns the model storage policy associated with the pipeline.

        Args:
            pipeline_id: the id of the pipeline, from which the policy is taken.

        Returns:
            ModelStoragePolicy: the model storage policy of the pipeline.
        """

        with MetadataDatabaseConnection(self._modyn_config) as database:
            pipeline: Pipeline = database.session.query(Pipeline).get(pipeline_id)

        policy = ModelStoragePolicy(
            self._ftp_dir,
            pipeline.full_model_strategy_name,
            pipeline.full_model_strategy_zip,
            pipeline.full_model_strategy_zip_algorithm,
            pipeline.full_model_strategy_config,
        )

        if pipeline.inc_model_strategy_name is not None:
            policy.register_incremental_model_strategy(
                pipeline.inc_model_strategy_name,
                pipeline.inc_model_strategy_zip,
                pipeline.inc_model_strategy_zip_algorithm,
                pipeline.inc_model_strategy_config,
                pipeline.full_model_interval,
            )

        return policy

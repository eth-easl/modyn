import gc
import json
import logging
import pathlib

import torch

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Pipeline, TrainedModel
from modyn.model_storage.internal.utils import ModelStoragePolicy
from modyn.utils import current_time_millis, dynamic_module_import

logger = logging.getLogger(__name__)


class ModelStorageManager:
    """Class used as manager of the model storage component.

    Implements all model storage related functionalities.
    """

    def __init__(self, modyn_config: dict, storage_dir: pathlib.Path, ftp_dir: pathlib.Path):
        """Constructor of the model storage manager. It establishes a
        connection to the metadata database in order to store information
        related to the trained models.

        Args:
            modyn_config: the modyn configuration.
            storage_dir: path to the folder, in which the trained models are stored.
            ftp_dir: FTP directory, which is used as temporary folder for serving trained models.
        """
        self._modyn_config = modyn_config
        self._storage_dir = storage_dir
        self._ftp_dir = ftp_dir

    def store_model(self, pipeline_id: int, trigger_id: int, checkpoint_path: pathlib.Path) -> int:
        """Store the trained model contained in the checkpoint file to disk. It
        uses the model storage policy that is specified for the pipeline.
        Depending on the trigger id, it is either stored fully (according to
        full model strategy) or incrementally by using the incremental model
        strategy.

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
        del state_dict
        del checkpoint["model"]

        # now checkpoint only contains optimizer state and metadata.
        local_metadata_filename = f"{current_time_millis()}_{pipeline_id}_{trigger_id}.metadata.zip"
        metadata_path = self._storage_dir / local_metadata_filename
        torch.save(checkpoint, metadata_path)
        del checkpoint
        self._clear_cuda_mem()

        # add the new model to the database.
        with MetadataDatabaseConnection(self._modyn_config) as database:
            return database.add_trained_model(
                pipeline_id, trigger_id, local_model_filename, local_metadata_filename, parent_id
            )

    def _clear_cuda_mem(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _handle_new_model(
        self,
        pipeline_id: int,
        trigger_id: int,
        state_dict: dict,
        model_path: pathlib.Path,
        policy: ModelStoragePolicy,
    ) -> int | None:
        """Handle the new model according to the model storage policy.

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
            parent_model_id: int | None = self._determine_parent_model_id(pipeline_id, trigger_id)
            if parent_model_id is not None:
                # load model state of the parent model.
                parent_model_state = self._reconstruct_model_state(parent_model_id, policy)

                # finally store the model delta.
                policy.incremental_model_strategy.store_model(state_dict, parent_model_state, model_path)

                del parent_model_state
                del state_dict
                self._clear_cuda_mem()

                return parent_model_id
            logger.warning("Previous model is not available! Storing full model...")

        # store the model in its entirety.
        policy.full_model_strategy.store_model(state_dict, model_path)

        del state_dict
        self._clear_cuda_mem()
        return None

    def _reconstruct_model_state(self, model_id: int, policy: ModelStoragePolicy) -> dict:
        """Reconstruct a given model according to the model storage policy. The
        function recursively calls itself whenever the model is stored as a
        delta. Otherwise it is stored according to a full model strategy and
        the model state can be retrieved.

        Args:
            model_id: the identifier of the model to be reconstructed.
            policy: the model storage policy of the pipeline.
        Returns:
            dict: the reconstructed model state. Refers to the same object as model_state.
        """

        # we recursively overwrite the model state.
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model: TrainedModel = database.session.get(TrainedModel, model_id)
        if not model.parent_model:
            # base case: we can load a fully stored model.
            model_state = self._get_base_model_state(model.pipeline_id)
            self._clear_cuda_mem()
            return policy.full_model_strategy.load_model(model_state, self._storage_dir / model.model_path)

        # recursive step: we recurse to load the model state of the parent model.
        model_state = self._reconstruct_model_state(model.parent_model, policy)

        self._clear_cuda_mem()

        # we apply the incremental strategy to load our model state.
        assert policy.incremental_model_strategy is not None
        return policy.incremental_model_strategy.load_model(model_state, self._storage_dir / model.model_path)

    def _get_base_model_state(self, pipeline_id: int) -> dict:
        """Get a randomly initialized model associated with the pipeline.

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
        # TODO(create issue): remove cuda and fix GPU loading for DLRM (also apex for model storage)
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        return model_handler(json.loads(model_config), device, amp).model.state_dict()

    def _determine_parent_model_id(self, pipeline_id: int, trigger_id: int) -> int | None:
        """Determines the id of the parent model given the trigger id of a
        pipeline. Usually, the last fully stored model is identified as such.
        The function returns None whenever no parent model can be found.

        Args:
            pipeline_id: the pipeline that generated the model.
            trigger_id: the trigger associated with the model.

        Returns:
            Optional[int]: the parent model id (if it can be found).
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            previous_model: TrainedModel = (
                database.session.query(TrainedModel)
                .filter(TrainedModel.pipeline_id == pipeline_id, TrainedModel.trigger_id == trigger_id - 1)
                .first()
            )

        # whenever the previous model is not present, a parent model cannot be determined.
        if not previous_model:
            return None
        # return the id of the previous model if its stored in its entirety.
        if previous_model.parent_model is None:
            return previous_model.model_id
        # otherwise return the parent model of the previous model.
        return previous_model.parent_model

    def load_model(self, model_id: int, metadata: bool) -> dict | None:
        """Loads a given model and optionally, also appends the metadata.

        Args:
            model_id: the model identifier of the model.
            metadata: whether metadata should be loaded alongside.

        Returns:
            Optional[dict]: dictionary containing the model state and metadata if the model exists.
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model: TrainedModel | None = database.session.get(TrainedModel, model_id)
            if model is None:
                logger.error(f"Model {model_id} does not exist.")
                return None
        policy = self.get_model_storage_policy(model.pipeline_id)

        # retrieve the model by loading its state dictionary.
        model_state = self._reconstruct_model_state(model_id, policy)
        model_dict = {"model": model_state}

        # append the metadata to the dictionary if specified.
        if metadata:
            metadata_dict = torch.load(self._storage_dir / model.metadata_path)
            model_dict.update(metadata_dict)
            del metadata_dict
            self._clear_cuda_mem()

        return model_dict

    def delete_model(self, model_id: int) -> bool:
        """Deletes a given model id. Only works, if all depending models
        (children) are deleted.

        Args:
            model_id: the identifier of the model.

        Returns:
            bool: True, whenever deletion was successful.
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            model: TrainedModel | None = database.session.get(TrainedModel, model_id)

            if model is None:
                logger.error(f"Trained model {model_id} was not found.")
                return False

            children = model.children
            if len(children) > 0:
                child_ids = [str(child.model_id) for child in children]
                logger.info(f"Model {model_id} has depending child models: {', '.join(child_ids)}")
                return False

            (self._storage_dir / model.model_path).unlink()
            (self._storage_dir / model.metadata_path).unlink()

            database.session.delete(model)
            database.session.commit()
        logger.info(f"Successfully deleted model {model_id}.")
        return True

    def get_model_storage_policy(self, pipeline_id: int) -> ModelStoragePolicy:
        """Returns the model storage policy associated with the pipeline.

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

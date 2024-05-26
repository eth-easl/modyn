from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.selector.internal.selector_strategies.utils import get_trigger_dataset_size
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.selector.internal.storage_backend.database import DatabaseStorageBackend


class RHOLossDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)

        self.remote_downsampling_strategy_name = "RemoteRHOLossDownsampling"

    def inform_next_trigger(self, next_trigger_id: int, selector_storage_backend: AbstractStorageBackend) -> None:
        if not isinstance(selector_storage_backend, DatabaseStorageBackend):
            raise ValueError("RHOLossDownsamplingStrategy requires a DatabaseStorageBackend")

        # The logic to train an IL model will be implemented here
        # Step 1: Fetch or create the rho_pipeline_id from metadata database
        #
        # Will create a new table (pipeline_id, rho_pipeline_id) representing the mapping
        # between pipeline_id and rho_pipeline_id, primary key being pipeline_id.
        #
        # Step 2: Prepare the training data for the IL model, by randomly sampling a predefined ratio of samples
        # from next_trigger_id's data and storing them as a TSS with identifier (rho_pipeline_id, next_trigger_id).

        # Step 3: Issue training request to the trainer server, with pipeline_id as rho_pipeline_id and trigger_id
        # as next_trigger_id. Wait for the training to complete. Store the model. Record model id in
        # downsampling_params, so that it can be fetched and used for downsampling.
        raise NotImplementedError

    def _prepare_holdout_set(
            self, next_trigger_id: int,
            rho_pipeline_id: int,
            selector_storage_backend: AbstractStorageBackend
    ) -> None:
        current_trigger_dataset_size = get_trigger_dataset_size(
            selector_storage_backend, self._pipeline_id, next_trigger_id, tail_triggers=0
        )

        raise NotImplementedError

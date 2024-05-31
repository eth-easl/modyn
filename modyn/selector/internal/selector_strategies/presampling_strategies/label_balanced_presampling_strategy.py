from modyn.config import PresamplingConfig
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractBalancedPresamplingStrategy
from modyn.selector.internal.storage_backend.abstract_storage_backend import AbstractStorageBackend


class LabelBalancedPresamplingStrategy(AbstractBalancedPresamplingStrategy):
    def __init__(
        self,
        presampling_config: PresamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        storage_backend: AbstractStorageBackend,
    ):
        super().__init__(presampling_config, modyn_config, pipeline_id, storage_backend, SelectorStateMetadata.label)

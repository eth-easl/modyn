from modyn.config import CoresetSelectionConfig
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from modyn.selector.internal.storage_backend.abstract_storage_backend import AbstractStorageBackend
from modyn.utils import instantiate_class


def instantiate_presampler(
    config: CoresetSelectionConfig, modyn_config: dict, pipeline_id: int, storage_backend: AbstractStorageBackend
) -> AbstractPresamplingStrategy:
    presampling_strategy = config.presampling_config.strategy
    presampling_config = config.presampling_config

    presampling_class = instantiate_class(
        "modyn.selector.internal.selector_strategies.presampling_strategies",
        presampling_strategy + "PresamplingStrategy",
        presampling_config,
        modyn_config,
        pipeline_id,
        storage_backend,
    )
    return presampling_class

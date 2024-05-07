import json
import logging
import pathlib
from typing import Optional, Union

from modyn.model_storage.internal.storage_strategies.full_model_strategies import AbstractFullModelStrategy
from modyn.model_storage.internal.storage_strategies.incremental_model_strategies import (
    AbstractIncrementalModelStrategy,
)
from modyn.utils import dynamic_module_import

logger = logging.getLogger(__name__)

FULL_MODEL_STRATEGY_MODULE = "modyn.model_storage.internal.storage_strategies.full_model_strategies"
INCREMENTAL_MODEL_STRATEGY_MODULE = "modyn.model_storage.internal.storage_strategies.incremental_model_strategies"


class ModelStoragePolicy:
    """
    Class used to represent the model storage policy. It loads the specified strategies.
    """

    def __init__(
        self,
        zipping_dir: pathlib.Path,
        full_model_strategy_name: str,
        full_model_strategy_zip: Optional[bool],
        full_model_strategy_zip_algorithm: Optional[str],
        full_model_strategy_config: Optional[str],
    ) -> None:
        self.zipping_dir = zipping_dir
        storage_strategy = self._setup_model_storage_strategy(
            full_model_strategy_name,
            full_model_strategy_zip,
            full_model_strategy_zip_algorithm,
            full_model_strategy_config,
            FULL_MODEL_STRATEGY_MODULE,
        )
        assert isinstance(storage_strategy, AbstractFullModelStrategy)
        self.full_model_strategy: AbstractFullModelStrategy = storage_strategy

        self.incremental_model_strategy: Optional[AbstractIncrementalModelStrategy] = None
        self.full_model_interval: Optional[int] = None

    def register_incremental_model_strategy(
        self,
        name: str,
        zip_enabled: Optional[bool],
        zip_algorithm: Optional[str],
        config: Optional[str],
        full_model_interval: Optional[int],
    ) -> None:
        setup_strategy = self._setup_model_storage_strategy(
            name, zip_enabled, zip_algorithm, config, INCREMENTAL_MODEL_STRATEGY_MODULE
        )
        assert isinstance(setup_strategy, AbstractIncrementalModelStrategy)
        self.incremental_model_strategy = setup_strategy
        if full_model_interval is not None:
            self._validate_full_model_interval(full_model_interval)

    def _validate_full_model_interval(self, full_model_interval: int) -> None:
        if full_model_interval <= 0:
            raise ValueError("Full model interval should be positive.")
        self.full_model_interval = full_model_interval

    def _setup_model_storage_strategy(
        self,
        name: str,
        zip_enabled: Optional[bool],
        zip_algorithm: Optional[str],
        config: Optional[str],
        module_name: str,
    ) -> Union[AbstractFullModelStrategy, AbstractIncrementalModelStrategy]:
        model_storage_module = dynamic_module_import(module_name)
        if not hasattr(model_storage_module, name):
            raise NotImplementedError(f"Strategy {name} not implemented!")
        model_storage_strategy_handler = getattr(model_storage_module, name)
        strategy_config = json.loads(config) if config else {}
        return model_storage_strategy_handler(
            self.zipping_dir, zip_enabled or False, zip_algorithm or "ZIP_DEFLATED", strategy_config
        )

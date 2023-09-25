import pathlib
from abc import ABC, abstractmethod
from zipfile import ZIP_DEFLATED

from modyn.utils import dynamic_module_import


class AbstractModelStorageStrategy(ABC):
    """
    Base class for all model storage strategies.
    """

    def __init__(self, zipping_dir: pathlib.Path, zip_activated: bool, zip_algorithm_name: str, config: dict):
        """
        Initialize a model storage strategy.

        Args:
            zipping_dir: directory, in which the model is zipped.
            zip_activated: whether the generated file is zipped.
            zip_algorithm_name: name of the zip algorithm.
            config: configuration options for the strategy.
        """
        self.zipping_dir = zipping_dir
        self.zip = zip_activated
        self.zip_algorithm = ZIP_DEFLATED
        self._validate_zip_config(zip_algorithm_name)

        self.validate_config(config)

    @abstractmethod
    def validate_config(self, config: dict) -> None:
        """
        Validates the strategy-dependent configuration options.

        Args:
            config: the configuration options.
        """
        raise NotImplementedError()

    def _validate_zip_config(self, zip_algorithm_name: str) -> None:
        if self.zip and zip_algorithm_name:
            zip_module = dynamic_module_import("zipfile")
            if not hasattr(zip_module, zip_algorithm_name):
                raise NotImplementedError(f"The zip algorithm {zip_algorithm_name} is unknown!")
            self.zip_algorithm = getattr(zip_module, zip_algorithm_name)

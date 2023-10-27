import pathlib
import tempfile
from abc import ABC, abstractmethod

from modyn.model_storage.internal.storage_strategies.abstract_model_storage_strategy import AbstractModelStorageStrategy
from modyn.utils import unzip_file, zip_file


class AbstractFullModelStrategy(AbstractModelStorageStrategy, ABC):
    """
    This is the base class for all full model strategies. That is, strategies which contain full information about
    a model in order to reproduce its model state.
    """

    @abstractmethod
    def _store_model(self, model_state: dict, file_path: pathlib.Path) -> None:
        """
        Stores the model state to the given file.

        Args:
            model_state: the state dictionary of the model.
            file_path: the path to the file in which to store the state.
        """
        raise NotImplementedError()

    def store_model(self, model_state: dict, file_path: pathlib.Path) -> None:
        if self.zip:
            with tempfile.NamedTemporaryFile(dir=self.zipping_dir) as temporary_file:
                temp_file_path = pathlib.Path(temporary_file.name)
                self._store_model(model_state, temp_file_path)
                zip_file(temp_file_path, file_path, self.zip_algorithm, remove_file=False)
        else:
            self._store_model(model_state, file_path)

    @abstractmethod
    def _load_model(self, base_model_state: dict, file_path: pathlib.Path) -> dict:
        """
        Load the model state from the given file.

        Args:
            base_model_state: the base model state which must be overwritten.
            file_path: the path to the file that contains the state information.

        Returns:
            dict: the state dictionary of the loaded model.
        """
        raise NotImplementedError()

    def load_model(self, base_model_state: dict, file_path: pathlib.Path) -> dict:
        if self.zip:
            with tempfile.NamedTemporaryFile(dir=self.zipping_dir) as temporary_file:
                temp_file_path = pathlib.Path(temporary_file.name)
                unzip_file(file_path, temp_file_path, compression=self.zip_algorithm, remove_file=False)
                return self._load_model(base_model_state, temp_file_path)
        return self._load_model(base_model_state, file_path)

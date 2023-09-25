import pathlib
import tempfile
from abc import ABC, abstractmethod

from modyn.model_storage.internal.storage_strategies.abstract_model_storage_strategy import AbstractModelStorageStrategy
from modyn.utils import unzip_file, zip_file


class AbstractIncrementalModelStrategy(AbstractModelStorageStrategy, ABC):
    """
    This is the base class for all incremental model strategies. These strategies build on the idea of storing a delta
    between two successive models in order to reproduce the latter one.
    """

    @abstractmethod
    def _store_model(self, model_state: dict, prev_model_state: dict, file_path: pathlib.Path) -> None:
        """
        Stores the delta between two successive models.

        Args:
            model_state: the newer model state.
            prev_model_state: the state of the preceding model.
            file_path: the path to the file in which the delta is stored.
        """
        raise NotImplementedError()

    def store_model(self, model_state: dict, prev_model_state: dict, file_path: pathlib.Path) -> None:
        if self.zip:
            with tempfile.NamedTemporaryFile(dir=self.zipping_dir) as temporary_file:
                temp_file_path = pathlib.Path(temporary_file.name)
                self._store_model(model_state, prev_model_state, temp_file_path)
                zip_file(temp_file_path, file_path, self.zip_algorithm, remove_file=False)
        else:
            self._store_model(model_state, prev_model_state, file_path)

    @abstractmethod
    def _load_model(self, prev_model_state: dict, file_path: pathlib.Path) -> None:
        """
        Loads a model state by overwriting the state of the preceding model.

        Args:
            prev_model_state: the state of the preceding model.
            file_path: the path to the file which contains the delta.
        """
        raise NotImplementedError()

    def load_model(self, prev_model_state: dict, file_path: pathlib.Path) -> None:
        if self.zip:
            with tempfile.NamedTemporaryFile(dir=self.zipping_dir) as temporary_file:
                temp_file_path = pathlib.Path(temporary_file.name)
                unzip_file(file_path, temp_file_path, compression=self.zip_algorithm, remove_file=False)
                self._load_model(prev_model_state, temp_file_path)
        else:
            self._load_model(prev_model_state, file_path)

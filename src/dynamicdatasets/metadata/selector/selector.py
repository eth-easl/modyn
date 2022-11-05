import sqlite3
from abc import abstractmethod

from .. import Metadata


class Selector(Metadata):
    def __init__(self, config: dict):
        super().__init__(config)

    @abstractmethod
    def get_next_batch(self) -> str:
        """
        Get the next batch as defined by the selector

        Returns:
            str: filename of the next ready batch
        """
        raise NotImplementedError

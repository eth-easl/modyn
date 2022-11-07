from abc import abstractmethod

from .. import Metadata


class Selector(Metadata):
    def __init__(self, config: dict):
        super().__init__(config)

    def get_next_batch(self) -> dict[str, list[int]]:
        """
        Return the next batch available

        Returns:
            dict[str, list[int]]: map of filename to indexes of the rows in the batch
        """
        return self._get_next_batch()

    @abstractmethod
    def _get_next_batch(self) -> dict[str, list[int]]:
        """
        Get the next batch as defined by the selector

        Returns:
            dict[str, list[int]]: map of filename to indexes of the rows in the batch
        """
        raise NotImplementedError

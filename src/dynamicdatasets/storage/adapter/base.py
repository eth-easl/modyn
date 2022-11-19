from abc import abstractmethod, ABC


class BaseAdapter(ABC):

    def __init__(self, config: dict):
        self._config = config

    @abstractmethod
    def get(self, keys: list[str]) -> list[bytes]:
        """
        Get data from the storage

        Args:
            key (str): key of the data

        Returns:
            dict: data
        """
        raise NotImplementedError

    @abstractmethod
    def put(self, key: list[str], data: list[bytes]) -> None:
        """
        Put data into the storage

        Args:
            key (str): key of the data
            data (bytes): data to store
        """
        raise NotImplementedError

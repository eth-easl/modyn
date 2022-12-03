from abc import abstractmethod, ABC


class BaseAdapter(ABC):

    def __init__(self, config: dict):
        self._config = config

    @abstractmethod
    def get(self, keys: list[str]) -> list[str]:
        """
        Get data from the storage

        Args:
            key (str): key of the data

        Returns:
            dict: data
        """
        raise NotImplementedError

    @abstractmethod
    def put(self, key: list[str], data: list[str]) -> None:
        """
        Put data into the storage

        Args:
            key (str): key of the data
            data (bytes): data to store
        """
        raise NotImplementedError

    @abstractmethod
    def query(self) -> list[str]:
        """
        Query the storage for new keys

        Returns:
            list[str]: list of keys
        """
        raise NotImplementedError

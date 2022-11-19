from abc import abstractmethod, ABC


class BaseAdapter(ABC):

    def __init__(self, config: dict) -> None:
        self._config = config

    @abstractmethod
    def get_next(self) -> list[bytes]:
        """
        Get next data from the data source

        Returns:
            bytes: data
        """
        raise NotImplementedError

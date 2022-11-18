from abc import abstractmethod


class Storable:

    @abstractmethod
    def store_task(self) -> str:
        """
        Get the select statement to get the rows of a batch

        Returns:
            str: select statement
        """
        raise NotImplementedError

    def __len__(self):
        pass

    def __getitem__(self, i):
        pass

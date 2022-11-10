from abc import abstractmethod


class Queryable:

    @abstractmethod
    def query_next(self) -> str:
        """
        Get the select statement to get the rows of a batch

        Returns:
            str: select statement
        """
        raise NotImplementedError

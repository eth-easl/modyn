from abc import abstractmethod


class Preprocessor:

    @abstractmethod
    def preprocess() -> str:
        """
        Get the select statement to get the rows of a batch

        Returns:
            str: select statement
        """
        raise NotImplementedError

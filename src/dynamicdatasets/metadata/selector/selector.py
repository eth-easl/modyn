from abc import abstractmethod

from .. import Metadata


class Selector(Metadata):
    def __init__(self, config: dict):
        super().__init__(config)

    def get_next_training_set(self) -> dict[str, list[int]]:
        """
        Return the next training_set available

        Returns:
            dict[str, list[int]]: map of filename to indexes of the sample in the training_set
        """
        return self._get_next_training_set()

    def _get_next_training_set(self) -> dict[str, list[int]]:
        """
        Get the next training_set as defined by the selector

        Returns:
            dict[str, list[int]]: map of filename to indexes of the sample in the training_set
        """
        cur = self._con.cursor()
        cur.execute(self._get_select_statement())
        sample = cur.fetchall()[0]
        self._update_training_set_metadata(sample[0], sample[1], 0)
        cur.execute(
            self._select_statement,
            (sample[0],
             ))
        sample = cur.fetchall()
        sample_dict = dict()
        for x, y in sample:
            sample_dict.setdefault(x, []).append(y)
        return sample_dict

    @abstractmethod
    def _get_select_statement(self) -> str:
        """
        Get the select statement to get the sample of a training_set

        Returns:
            str: select statement
        """
        raise NotImplementedError

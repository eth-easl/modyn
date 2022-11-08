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

    def _get_next_batch(self) -> dict[str, list[int]]:
        """
        Get the next batch as defined by the selector

        Returns:
            dict[str, list[int]]: map of filename to indexes of the rows in the batch
        """
        cur = self._con.cursor()
        cur.execute(self._get_select_statement())
        row = cur.fetchall()[0]
        self._update_batch_metadata(row[0], row[1], 0)
        cur.execute(
            self._select_statement,
            (row[0],
             ))
        rows = cur.fetchall()
        row_dict = dict()
        for x, y in rows:
            row_dict.setdefault(x, []).append(y)
        return row_dict

    @abstractmethod
    def _get_select_statement(self) -> str:
        """
        Get the select statement to get the rows of a batch

        Returns:
            str: select statement
        """
        raise NotImplementedError

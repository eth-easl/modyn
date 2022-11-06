
from . import Selector


class LastSelector(Selector):

    def _get_next_batch(self) -> dict[str, list[int]]:
        """
        Return the last new batch available

        Returns:
            dict[str, list[int]]: map of filename to indexes of the rows in the batch
        """
        cur = self._con.cursor()
        cur.execute(
            'SELECT id, score FROM batch_metadata WHERE new=1 ORDER BY timestamp DESC LIMIT 1')
        row = cur.fetchall()[0]
        self._update_batch_metadata(row[0], row[1], 0)
        cur.execute(
            'SELECT filename, row FROM row_metadata WHERE batch_id=? ORDER BY row ASC',
            (row[0],
             ))
        rows = cur.fetchall()
        row_dict = dict()
        for x, y in rows:
            row_dict.setdefault(x, []).append(y)
        return row_dict

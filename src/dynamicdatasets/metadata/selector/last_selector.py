
from . import Selector


class LastSelector(Selector):

    def get_next_batch(self) -> str:
        cur = self._con.cursor()
        cur.execute(
            'SELECT id, filename, score FROM batch_metadata WHERE new=1 ORDER BY timestamp ASC LIMIT 1')
        row = cur.fetchall()[0]
        self.update_batch_metadata(row[0], row[2], 0)
        return row[1]

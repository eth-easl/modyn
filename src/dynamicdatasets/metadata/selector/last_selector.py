
from . import Selector


class LastSelector(Selector):
    _select_statement = '''SELECT filename, row_metadata.row
                           FROM row_metadata
                           JOIN batch_to_row ON row_metadata.row = batch_to_row.row
                           WHERE batch_id=%s ORDER BY row_metadata.row ASC'''

    def _get_select_statement(self) -> str:
        return 'SELECT id, score FROM batch_metadata WHERE new=1 ORDER BY timestamp DESC LIMIT 1'

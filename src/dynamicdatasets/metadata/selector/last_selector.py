
from . import Selector


class LastSelector(Selector):
    _select_statement = '''SELECT filename, sample_metadata.sample
                           FROM sample_metadata
                           JOIN training_set_to_sample ON sample_metadata.sample = training_set_to_sample.sample
                           WHERE training_set_id=%s ORDER BY sample_metadata.sample ASC'''

    def _get_select_statement(self) -> str:
        return 'SELECT id, score FROM training_set_metadata WHERE new=1 ORDER BY timestamp DESC LIMIT 1'

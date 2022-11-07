from abc import ABC, abstractmethod

from .. import Metadata


class Scorer(Metadata):
    """
    Abstract class to score the samples according to the implementors defined data importance measure
    and update the corresponding metadata database
    """
    _con = None
    _nr_batches = 0
    _batch_selection = None
    _row_selection = None

    def __init__(self, config: dict):
        """
        Args:
            config (dict): YAML config file with the required structure.

            See src/config/README.md for more information

            data_storage (DataStorage): Data storage instance to store the newly created data instances
        """
        super().__init__(config)

    def _add_batch(
            self,
            filename: str,
            rows: list[int],
            initial=True,
            batch_id=None) -> int:
        """
        Add a batch to the data scorer metadata database

        Calculate the batch score by the median of the row scores

        Args:
            filename (str): filename of the batch
            rows (list[int]): row numbers in the batch
            initial (bool): if adding a batch initially or otherwise

        Returns:
            int: batch_id of the added batch
        """
        self._nr_batches += 1
        if (initial and self._nr_batches %
                self._config['metadata']['nr_files_update'] == 0):
            self._create_shuffled_batches(
                self._config['metadata']['nr_files_update'],
                self._config['feeder']['batch_size'])
        if batch_id is None:
            batch_id = self._add_batch_to_metadata()

        scores = []
        for row in rows:
            score = self._get_score()
            self._add_row_to_metadata(row, batch_id, score, filename)
            scores.append(score)
        median = self._get_cumulative_score(scores)
        self._update_batch_metadata(batch_id, median, 1)
        return batch_id

    def _create_shuffled_batches(
            self, batch_count: int, batch_size: int):
        """
        Update an existing batch based on a batch and row selection criterion. Select a number of batches
        and decide on a total number of samples for the new batch. The created batch will contain equal
        proportion of the selected batches.

        Args:
            batch_selection (str, optional): sql to select batches according to a criteria.
            row_selection (str, optional): sql to select rows accordig to a criteria.
            batch_count (int, optional): number of batches to include in the updated batch.
            batch_size (int, optional): number of rows in the resulting batch.
        """
        batches = self.fetch_batches(self._batch_selection, batch_count)
        row_count = int(batch_size / batch_count)

        new_batch_id = None

        for batch_id in batches:
            rows = self._fetch_rows(self._row_selection, row_count, batch_id)

            for filename in rows:
                bid = self._add_batch(
                    filename,
                    rows[filename],
                    initial=False,
                    batch_id=new_batch_id)

                if new_batch_id is None:
                    new_batch_id = bid

    def add_batch(
            self, filename: str, rows: list[int]):
        """
        Add a batch to the data scorer metadata database

        Args:
            filename (str): filename of the batch
            rows (list[int]): row numbers in the batch

        Returns:
            int: batch_id of the added batch
        """
        return self._add_batch(filename, rows)

    @abstractmethod
    def _get_score(self) -> float:
        """
        Generate score according to data importance

        Returns:
            float: score
        """
        raise NotImplementedError

    @abstractmethod
    def _get_cumulative_score(self, scores: list[int]):
        """
        Calculate the cumulative score of a list of scores

        Args:
            scores (list[int]): list of scores

        Returns:
            float: cumulative score
        """
        raise NotImplementedError

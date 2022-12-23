from abc import ABC, abstractmethod
import psycopg2


class Selector(ABC):

    _config: dict = dict()
    _con = None

    _create_trainings_table_sql = '''CREATE TABLE IF NOT EXISTS trainings (
            id SERIAL PRIMARY KEY,
            training_set_size INTEGER NOT NULL,
            num_workers INTEGER NOT NULL
        );'''
    _create_training_samples_table_sql = '''CREATE TABLE IF NOT EXISTS training_samples (
            training_id INTEGER,
            training_set_number INTEGER,
            sample_number INTEGER,
            sample_key VARCHAR(255) NOT NULL,
            FOREIGN KEY (training_id) REFERENCES trainings(id),
            CONSTRAINT unique_training_sample PRIMARY KEY (training_id,training_set_number, sample_number)
        );'''

    _insert_training_sql = '''INSERT INTO trainings(training_set_size, num_workers) VALUES(%s,%s) RETURNING id;'''
    _fetch_training_samples_sql = '''SELECT sample_key FROM training_samples
                                     WHERE training_id = %s and training_set_number = %s order by sample_number;'''
    _fetch_training_info_sql = '''SELECT training_set_size, num_workers FROM trainings where id = %s'''

    def __init__(self, config: dict):
        self._config = config
        self._con = psycopg2.connect(
            host=config['selector']['postgresql']['host'],
            port=config['selector']['postgresql']['port'],
            user=config['selector']['postgresql']['user'],
            password=config['selector']['postgresql']['password']
        )
        self._setup_database()

    def _setup_database(self) -> None:
        """
        Ensure the tables required are created in the DB
        """
        assert self._con is not None, "No connection established"
        cur = self._con.cursor()
        cur.execute(self._create_trainings_table_sql)
        cur.execute(self._create_training_samples_table_sql)
        self._con.commit()

    @abstractmethod
    def _select_new_training_samples(
            self,
            training_id: int,
            training_set_size: int
    ) -> list[str]:
        """
        Selects a new training set of samples for the given training id. Samples should be selected from
        the metadata service

        Returns:
            list(str): the training sample keys for the newly selected training_set
        """
        raise NotImplementedError

    def _insert_training_samples(
            self,
            training_samples: list[str],
            training_id: int,
            training_set_number: int) -> None:
        """
        Insert the list of training_samples into the DB
        """

        assert self._con is not None, "No connection established"

        # Form the sql query
        insert_samples_sql = 'INSERT INTO training_samples ' + \
                             '(training_id, training_set_number, sample_number, sample_key) VALUES '

        for idx, sample_key in enumerate(training_samples):
            value_list = "(%s,%s,%s,'%s')," % (
                training_id, training_set_number, idx, sample_key)
            insert_samples_sql = insert_samples_sql + value_list

        # replace last , with ;
        insert_samples_sql = insert_samples_sql[:-1] + ";"
        cur = self._con.cursor()
        cur.execute(insert_samples_sql)
        self._con.commit()

    def _prepare_training_set(
            self,
            training_id: int,
            training_set_number: int,
            training_set_size: int,
    ) -> list[str]:
        """
        Create a new training set of samples for the given training id. New samples are selected from
        the select_new_samples method and are inserted into the database for the given set number.

        Returns:
            list(str): the training sample keys for the newly prepared training_set
        """
        training_samples = self._select_new_training_samples(training_id, training_set_size)

        # Throw error if no new samples are selected
        if (len(training_samples) == 0):
            raise ValueError("No new samples selected")

        self._insert_training_samples(training_samples, training_id, training_set_number)
        return training_samples

    def _get_training_set_partition(
            self,
            training_id: int,
            training_samples: list[str],
            worker_id: int) -> list[str]:
        """
        Return the required subset of training samples for the particular worker id
        The subset is calculated by taking an offset from the start based on the given worker id
        """

        training_set_size, num_workers = self._get_info_for_training(training_id)

        # TODO(#36): Handle training_set_size % num_workers > 0
        worker_subset_size = int(training_set_size / num_workers)
        start_index = worker_id * worker_subset_size
        training_samples_subset = training_samples[start_index: start_index +
                                                   worker_subset_size]
        return training_samples_subset

    def register_training(self, training_set_size: int,
                          num_workers: int) -> int:
        """
        Creates a new training object in the database with the given training_set_size and num_workers
        Returns:
            The id of the newly created training object
        """
        assert self._con is not None, "No connection established"

        cur = self._con.cursor()
        cur.execute(
            self._insert_training_sql,
            (training_set_size, num_workers)
        )
        training_set_id = cur.fetchone()
        self._con.commit()

        assert training_set_id is not None and training_set_id[0] is not None, "Insertion failed"

        return training_set_id[0]

    def _get_info_for_training(self, training_id: int) -> tuple[int, int]:
        """
        Queries the database for the the training set size and number of workers for a given training.

        Returns:
            Tuple of training set size and number of workers.
        """
        assert self._con is not None, "No connection established"
        cur = self._con.cursor()

        # Get the training_set_size and num_workers for this training_id
        cur.execute(self._fetch_training_info_sql, [training_id])
        training_info = cur.fetchone()
        if (training_info is None):
            raise Exception("Invalid training id")
        training_set_size, num_workers = training_info

        return training_set_size, num_workers

    def _fetch_training_set_if_exists(self, training_id: int, training_set_number: int) -> tuple[list[str], bool]:
        """
        For a given training_set and training_set_number, fetch the pre-calculated training set from
        the database, if it has been calculated.

        Returns:
            Tuple containing a list of keys describing the training set and a boolean to indicate whether the
            training set has been pre-calculated or the result is empty.
        """
        assert self._con is not None, "No connection established"
        cur = self._con.cursor()

        # Fetch the samples for that training and training set
        cur.execute(self._fetch_training_samples_sql,
                    (training_id, training_set_number))
        training_samples = cur.fetchall()

        if len(training_samples) > 0:
            return [x[0] for x in training_samples], True
        else:
            return [], False

    def _create_or_fetch_existing_set(self, training_id: int, training_set_number: int) -> list[str]:
        """
        For a given training_set and training_set_number, fetch the pre-calculated training set from
        the database. In case the set has not yet been calculated, calculate it.

        Returns:
            List of keys describing the training set with number `training_set_number` for the training
            with `training_id`
        """
        training_set_size, _ = self._get_info_for_training(training_id)
        training_samples, set_exists = self._fetch_training_set_if_exists(training_id, training_set_number)

        if not set_exists:
            training_samples = self._prepare_training_set(training_id, training_set_number, training_set_size)

        return training_samples

    def get_sample_keys(self, training_id: int,
                        training_set_number: int, worker_id: int) -> list[str]:
        """
        For a given training_id, training_set_number and worker_id, it returns a subset of sample
        keys so that the data can be queried from storage.

        Returns:
            List of keys for the samples to be returned to that particular worker
        """

        training_samples = self._create_or_fetch_existing_set(training_id, training_set_number)

        training_samples_subset = self._get_training_set_partition(
            training_id, training_samples, worker_id)

        return training_samples_subset

import psycopg2


class MetadataDatabase():
    """
    Store the metadata for all the training samples for a given training.
    """

    def __init__(self, config: dict):
        self.__config = config
        self.__con = psycopg2.connect(
            host=self.__config['metadata_database']['postgresql']['host'],
            port=self.__config['metadata_database']['postgresql']['port'],
            database=self.__config['metadata_database']['postgresql']['database'],
            user=self.__config['metadata_database']['postgresql']['user'],
            password=self.__config['metadata_database']['postgresql']['password']
        )
        self.__con.autocommit = False
        self.__cursor = self.__con.cursor()
        self.initialize_db()

    def initialize_db(self) -> None:
        """
        Create tables if they do not exist.
        """
        self.__cursor.execute(
            'CREATE TABLE IF NOT EXISTS metadata_database ('
            'id SERIAL PRIMARY KEY,'
            'key varchar(255) NOT NULL,'
            'score float NOT NULL,'
            'seen int NOT NULL,'
            'label int NOT NULL,'
            'data text NOT NULL,'
            'training_id int NOT NULL)'
        )
        self.__cursor.execute(
            'CREATE INDEX IF NOT EXISTS storage_key_idx ON metadata_database (key)'
        )
        self.__cursor.execute(
            'CREATE INDEX IF NOT EXISTS storage_training_id_idx ON metadata_database (training_id)'
        )
        self.__con.commit()

        self.__cursor.execute(
            'CREATE TABLE IF NOT EXISTS training_infos ('
            'id SERIAL PRIMARY KEY, '
            'training_id int NOT NULL'
            'num_workers int NOT NULL'
            'training_set_size int NOT NULL)'
        )

    def set(
            self,
            keys: list[str],
            scores: list[float],
            seens: list[bool],
            labels: list[int],
            datas: list[bytes],
            training_id: int) -> None:
        """
        Set the metadata for a given training. Will replace keys where they exist!

        Args:
            keys (list[str]): List of keys.
            scores (list[float]): List of scores.
            datas (list[bytes]): List of data.
            training_id (int): Training id.
        """
        self.__cursor.execute(
            "DELETE FROM metadata_database WHERE key IN %s AND training_id = %s",
            (tuple(keys),
             training_id))
        for key, score, seen, label, data in zip(keys, scores, seens, labels, datas):
            self.__cursor.execute(
                ("INSERT INTO metadata_database (key, score, seen, label, data, training_id)"
                    "VALUES (%s, %s, %s, %s, %s, %s)"),
                (key,
                 score,
                 seen,
                 label,
                 data,
                 training_id))
        self.__con.commit()

    def get_by_keys(
            self, keys: list[str], training_id: int) -> tuple[list[str], list[float], list[str]]:
        """
        Get the metadata for a given training and keys.

        Args:
            keys (list[str]): List of keys.
            training_id (int): Training id.

        Returns:
            list[tuple[str, float, str]]: List of keys, scores and data.
        """
        self.__cursor.execute(
            "SELECT key, score, seen, label, data FROM metadata_database WHERE key IN %s AND training_id = %s",
            (tuple(keys),
             training_id))
        data = self.__cursor.fetchall()
        return_keys = [d[0] for d in data]
        scores = [d[1] for d in data]
        seen = [d[2] for d in data]
        labels = [d[3] for d in data]
        return_data = [d[4] for d in data]
        return return_keys, scores, seen, labels, return_data

    def get_by_query(self, query: str) -> tuple[list[str], list[float], list[str]]:
        """
        Get the metadata for a given training and a executable query.

        Args:
            query (str): Executable query.

        Returns:
            list[tuple[str, float, str]]: List of keys, scores and data.
        """
        self.__cursor.execute(query)
        data = self.__cursor.fetchall()
        keys = [d[0] for d in data]
        scores = [d[1] for d in data]
        seen = [d[2] for d in data]
        labels = [d[3] for d in data]
        return_data = [d[4] for d in data]
        return keys, scores, seen, labels, return_data

    def get_keys_by_query(self, query: str) -> list[str]:
        """
        Get the keys for a given training and a executable query.

        Args:
            query (str): Executable query.

        Returns:
            list[str]: List of keys.
        """
        self.__cursor.execute(query)
        data = self.__cursor.fetchall()
        return_data: list[str] = [d[0] for d in data]
        return return_data

    def delete_training(self, training_id: int) -> None:
        """
        Delete the metadata for a given training.

        Args:
            training_id (int): Training id.
        """
        self.__cursor.execute(
            "DELETE FROM metadata_database WHERE training_id = %s", (training_id,))
        self.__con.commit()

    def register_training(self, training_set_size: int, num_workers: int) -> None:
        self.__cursor.execute(
            """INSERT INTO trainings(training_set_size, num_workers) VALUES(%s,%s) RETURNING id;""",
            (training_set_size, num_workers))
        training_set_id = self.__cursor.fetchone()
        self.__con.commit()
        return training_set_id

    def get_training_info(self, training_id: int) -> tuple[int, int]:
        self.__cursor.execute(
            "SELECT training_set_size, num_workers FROM training_infos WHERE training_id = %s", (training_id, ))
        data = self.__cursor.fetchall()
        assert len(data) == 1
        return data[0][0], data[0][1]

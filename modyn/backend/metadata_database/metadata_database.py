import psycopg2


class MetadataDatabase(object):
    """
    Store the metadata for all the training samples for a given training.
    """

    def __init__(self, config: dict):
        self.__config = config
        self.__con = psycopg2.connect(
            host=config['metadata_database']['postgresql']['host'],
            port=config['metadata_database']['postgresql']['port'],
            database=config['metadata_database']['postgresql']['database'],
            user=config['metadata_database']['postgresql']['user'],
            password=config['metadata_database']['postgresql']['password']
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
            score: list[float],
            data: list[bytes],
            training_id: int) -> None:
        """
        Set the optimal dataset metadata for a given training.

        Args:
            keys (list[str]): List of keys.
            score (list[float]): List of scores.
            data (list[bytes]): List of data.
            training_id (int): Training id.
        """
        self.__cursor.execute(
            "DELETE FROM metadata_database WHERE key IN %s AND training_id = %s",
            (tuple(keys),
             training_id))
        for i in range(len(keys)):
            self.__cursor.execute(
                "INSERT INTO metadata_database (key, score, data, training_id) VALUES (%s, %s, %s, %s)",
                (keys[i],
                 score[i],
                    data[i],
                    training_id))
        self.__con.commit()

    def get_by_keys(
            self, keys: list[str], training_id: int) -> tuple[list[str], list[float], list[str]]:
        """
        Get the optimal dataset metadata for a given training and keys.

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
        Get the optimal dataset metadata for a given training and a executable query.

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
        Delete the optimal dataset metadata for a given training.

        Args:
            training_id (int): Training id.
        """
        self.__cursor.execute(
            "DELETE FROM metadata_database WHERE training_id = %s", (training_id,))
        self.__con.commit()

    def register_training(self, training_id: int, training_set_size: int, num_workers: int) -> None:
        self.__cursor.execute(
            "INSERT INTO training_infos (training_id, num_workers, training_set_size) VALUES (%s, %s, %s)",
            (training_id, num_workers, training_set_size))

    def get_training_info(self, training_id: int) -> tuple[int, int]:
        self.__cursor.execute(
            "SELECT training_set_size, num_workers FROM training_infos WHERE training_id = %s", (training_id, ))
        data = self.__cursor.fetchall()
        assert len(data) == 1
        return data[0][0], data[0][1]

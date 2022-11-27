import psycopg2


class OptimalDatasetMetadata(object):
    """
    Store the metadata of the optimal dataset for a given training.
    """
    def __init__(self, config: dict):
        self.__config = config
        self.__con = psycopg2.connect(
            host=config['odm']['postgresql']['host'],
            port=config['odm']['postgresql']['port'],
            database=config['odm']['postgresql']['database'],
            user=config['odm']['postgresql']['user'],
            password=config['odm']['postgresql']['password']
        )
        self.__cursor = self.__con.cursor()
        self.create_table()

    def create_table(self) -> None:
        """
        Create tables if they do not exist.
        """
        self.__cursor.execute(
            'CREATE TABLE IF NOT EXISTS odm_storage ('
            'id SERIAL PRIMARY KEY,'
            'key varchar(255) NOT NULL,'
            'score float NOT NULL,'
            'data text NOT NULL,'
            'training_id int NOT NULL)'
        )
        self.__cursor.execute(
            'CREATE INDEX IF NOT EXISTS storage_key_idx ON odm_storage (key)'
        )
        self.__cursor.execute(
            'CREATE INDEX IF NOT EXISTS storage_training_id_idx ON odm_storage (training_id)'
        )
        self.__con.commit()

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
            "DELETE FROM odm_storage WHERE key IN %s AND training_id = %s",
            (tuple(keys),
             training_id))
        for i in range(len(keys)):
            self.__cursor.execute(
                "INSERT INTO odm_storage (key, score, data, training_id) VALUES (%s, %s, %s, %s)",
                (keys[i],
                 score[i],
                    data[i],
                    training_id))
        self.__con.commit()

    def get_by_keys(
            self, keys: list[str], training_id: int) -> list[tuple[str, float, str]]:
        """
        Get the optimal dataset metadata for a given training and keys.

        Args:
            keys (list[str]): List of keys.
            training_id (int): Training id.

        Returns:
            list[tuple[str, float, str]]: List of keys, scores and data.
        """
        self.__cursor.execute(
            "SELECT key, score, data FROM odm_storage WHERE key IN %s AND training_id = %s",
            (tuple(keys),
             training_id))
        data = self.__cursor.fetchall()
        data = map(list, zip(*data))
        return data[0], data[1], data[2]

    def get_by_query(self, query: str) -> list[tuple[str, float, str]]:
        """
        Get the optimal dataset metadata for a given training and a executable query.

        Args:
            query (str): Executable query.

        Returns:
            list[tuple[str, float, str]]: List of keys, scores and data.
        """
        self.__cursor.execute(query)
        data = self.__cursor.fetchall()
        data = map(list, zip(*data))
        return data[0], data[1], data[2]

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
        data = [d[0] for d in data]
        return data

    def delete_training(self, training_id: int) -> None:
        """
        Delete the optimal dataset metadata for a given training.

        Args:
            training_id (int): Training id.
        """
        self.__cursor.execute(
            "DELETE FROM odm_storage WHERE training_id = %s", (training_id,))
        self.__con.commit()

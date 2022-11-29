import psycopg2


class NewQueue(object):
    """
    New queue service is responsible for storing data in a database and
    retrieving data from the database upon request in a queue like fashion.

    Args:
        object (_type_): _description_
    """
    def __init__(self, config):
        self.__config = config
        self.__conn = psycopg2.connect(
            host=self.__config['newqueue']['postgresql']['host'],
            port=self.__config['newqueue']['postgresql']['port'],
            database=self.__config['newqueue']['postgresql']['database'],
            user=self.__config['newqueue']['postgresql']['user'],
            password=self.__config['newqueue']['postgresql']['password']
        )
        self.__cursor = self.__conn.cursor()
        self.create_table()

    def create_table(self):
        """
        Create tables if they do not exist.
        """
        self.__cursor.execute(
            'CREATE TABLE IF NOT EXISTS queue_data ('
            'id SERIAL PRIMARY KEY,'
            'key VARCHAR(255) NOT NULL UNIQUE,'
            'created TIMESTAMP NOT NULL,'
            'updated TIMESTAMP NOT NULL'
            ')'
        )
        self.__cursor.execute(
            'CREATE INDEX IF NOT EXISTS newqueue_key_idx ON queue_data (key)'
        )
        self.__cursor.execute(
            'CREATE TABLE IF NOT EXISTS queue ('
            'id SERIAL PRIMARY KEY,'
            'key VARCHAR(255) NOT NULL,'
            'training_id INTEGER NOT NULL,'
            'read TIMESTAMP NOT NULL,'
            'FOREIGN KEY (key) REFERENCES queue_data(key)'
            ')'
        )
        self.__cursor.execute(
            'CREATE INDEX IF NOT EXISTS queue_key_idx ON queue (key)'
        )
        self.__conn.commit()

    def add(self, keys: list[str]) -> None:
        """
        Add keys to the queue.

        Args:
            keys (list[str]): List of keys to add to the queue.
        """
        for key in keys:
            self.__cursor.execute(
                'INSERT INTO queue_data (key, created, updated) VALUES (%s, NOW(), NOW())',
                (key,)
            )
        self.__conn.commit()

    def get_next(self, limit: int, training_id: int) -> list[str]:
        """
        Get the next keys from the queue for a training id.

        Args:
            limit (int): Number of keys to retrieve. 
                         If limit is greater than the number of 
                         keys in the queue, all keys will be returned.
            training_id (int): Training id to retrieve keys for.

        Returns:
            list[str]: List of keys.
        """
        self.__cursor.execute(
            'SELECT key '
            'FROM queue_data '
            'WHERE key NOT IN'
            '(SELECT key FROM queue WHERE training_id = %s) '
            'ORDER BY created ASC '
            'LIMIT %s',
            (training_id, limit,)
        )
        keys = self.__cursor.fetchall()
        keys = [key[0] for key in keys]
        for key in keys:
            self.__cursor.execute(
                'INSERT INTO queue (key, training_id, read) VALUES (%s, %s, NOW())',
                (key, training_id)
            )
        self.__conn.commit()
        return keys

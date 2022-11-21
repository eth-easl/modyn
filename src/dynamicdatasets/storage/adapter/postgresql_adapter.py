import psycopg2

from .base import BaseAdapter


class PostgreSQLAdapter(BaseAdapter):
    """PostgresqlAdapter is a wrapper around the psycopg2 module, which provides a
    dictionary-like interface to a PostgreSQL database.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.__db = psycopg2.connect(
            host=self._config['storage']['postgresql']['host'],
            port=self._config['storage']['postgresql']['port'],
            database=self._config['storage']['postgresql']['database'],
            user=self._config['storage']['postgresql']['user'],
            password=self._config['storage']['postgresql']['password']
        )
        self.__cursor = self.__db.cursor()
        self.create_table()

    def create_table(self) -> None:
        self.__cursor.execute(
            "CREATE TABLE IF NOT EXISTS storage (key varchar(255) PRIMARY KEY, data TEXT)")
        self.__db.commit()

    def get(self, keys: list[str]) -> list[str]:
        self.__cursor.execute(
            "SELECT data FROM storage WHERE key IN %s", (tuple(keys),))
        data = self.__cursor.fetchall()
        data = [d[0] for d in data]
        if data is None:
            return None
        else:
            return data

    def put(self, key: list[str], data: list[str]) -> None:
        self.__cursor.execute(
            "INSERT INTO storage (key, data) VALUES (%s, %s)", (key, data))
        self.__db.commit()

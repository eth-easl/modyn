import psycopg2

from .base import BaseAdapter


class PostgreSQLAdapter(BaseAdapter):
    """PostgresqlAdapter is a wrapper around the psycopg2 module, which provides a
    dictionary-like interface to a PostgreSQL database.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.__con = psycopg2.connect(
            host=self._config['storage']['postgresql']['host'],
            port=self._config['storage']['postgresql']['port'],
            database=self._config['storage']['postgresql']['database'],
            user=self._config['storage']['postgresql']['user'],
            password=self._config['storage']['postgresql']['password']
        )
        self.__cursor = self.__con.cursor()
        self.create_table()

    def create_table(self) -> None:
        self.__cursor.execute(
            'CREATE TABLE IF NOT EXISTS storage ('
            'id SERIAL PRIMARY KEY,'
            'key varchar(255) UNIQUE NOT NULL,'
            'data TEXT NOT NULL, '
            'query_key boolean DEFAULT false);'
        )
        self.__cursor.execute(
            'CREATE INDEX IF NOT EXISTS storage_key_idx ON storage (key);'
        )
        self.__con.commit()

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
        for i in range(len(key)):
            self.__cursor.execute(
                'INSERT INTO storage (key, data) VALUES (%s, %s)',
                (key[i],
                 data[i]))
        self.__con.commit()

    def query(self) -> list[str]:
        self.__cursor.execute(
            'SELECT key FROM storage WHERE query_key = false')
        keys = self.__cursor.fetchall()
        keys = [k[0] for k in keys]
        self.__cursor.execute(
            'UPDATE storage SET query_key = true WHERE key IN %s',
            (tuple(keys),))
        return keys

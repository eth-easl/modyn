from sqlitedict import SqliteDict

from .base import BaseAdapter


class SQLiteAdapter(BaseAdapter):
    """SqliteAdapter is a wrapper around the sqlitedict module, which provides a
    dictionary-like interface to a SQLite database.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.__db = SqliteDict(
            self._config['storage']['sqlite']['path'],
            autocommit=True)

    def get(self, keys: list[str]) -> list[str]:
        data = []
        for key in keys:
            data.append(self.__db[key])
        return data

    def put(self, key: list[str], data: list[str]) -> None:
        for i in range(len(key)):
            self.__db[key[i]] = data[i]

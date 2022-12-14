import dbm

from .base import BaseAdapter


class DBMAdapter(BaseAdapter):
    """DbmAdapter is a wrapper around the dbm module, which provides a
    dictionary-like interface to a dbm database.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.__db = dbm.open(self._config['storage']['dbm']['path'], 'c')
        self.__db.close()

    def get(self, keys: list[str]) -> list[str]:
        self.__db = dbm.open(self._config['storage']['dbm']['path'], 'r')
        data = []
        for key in keys:
            data.append(self.__db[key])
        self.__db.close()
        # TODO(vGsteiger): We currently return a list of bytes here,
        #  but want to return a list of strings. How exactly do we convert that? We cannot do
        # str(x) for x in data, because then we would cast each byte individually.
        return data

    def put(self, key: list[str], data: list[str]) -> None:
        self.__db = dbm.open(self._config['storage']['dbm']['path'], 'c')
        for i in range(len(key)):
            self.__db[key[i]] = data[i]
        self.__db.close()

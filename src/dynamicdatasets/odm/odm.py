import psycopg2


class OptimalDatasetMetadata:
    def __init__(self, config: dict):
        self.__db = psycopg2.connect(
            host=config['odm']['postgresql']['host'],
            port=config['odm']['postgresql']['port'],
            user=config['odm']['postgresql']['user'],
            password=config['odm']['postgresql']['password']
        )
        self.__cursor = self.__db.cursor()
        self.create_table()

    def create_table(self) -> None:
        self.__cursor.execute(
            "CREATE TABLE IF NOT EXISTS storage (key varchar(255) PRIMARY KEY, score float, data bytea)")
        self.__db.commit()

    def set(
            self,
            keys: list[str],
            score: list[float],
            data: list[bytes]) -> None:
        self.__cursor.execute(
            "DELETE FROM storage WHERE key IN %s", (tuple(keys),))
        for i in range(len(keys)):
            self.__cursor.execute(
                "INSERT INTO storage (key, score, data) VALUES (%s, %s, %s)",
                (keys[i],
                 score[i],
                    data[i]))
        self.__db.commit()

    def get_by_keys(self, keys: list[str]) -> list[tuple[str, float, bytes]]:
        self.__cursor.execute(
            "SELECT key, score, data FROM storage WHERE key IN %s", (tuple(keys),))
        data = self.__cursor.fetchall()
        data = map(list, zip(*data))
        return data[0], data[1], data[2]

    def get_by_query(self, query: str) -> list[tuple[str, float, bytes]]:
        self.__cursor.execute(query)
        data = self.__cursor.fetchall()
        data = map(list, zip(*data))
        return data[0], data[1], data[2]

    def get_keys_by_query(self, query: str) -> list[str]:
        self.__cursor.execute(query)
        data = self.__cursor.fetchall()
        data = [d[0] for d in data]
        return data

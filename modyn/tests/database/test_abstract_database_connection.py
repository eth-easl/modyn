from modyn.database.abstract_database_connection import AbstractDatabaseConnection


def get_minimal_modyn_config() -> dict:
    return {
        "test_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "hostname": "",
            "port": 0,
            "database": ":memory:",
        }
    }


class TestAbstractDatabaseConnection(AbstractDatabaseConnection):
    __test__ = False

    def __init__(self, config):
        super().__init__(config)
        self.drivername = config["test_database"]["drivername"]
        self.username = config["test_database"]["username"]
        self.password = config["test_database"]["password"]
        self.host = config["test_database"]["hostname"]
        self.port = config["test_database"]["port"]
        self.database = config["test_database"]["database"]

    def create_tables(self):
        pass


def test_database_connection():
    with TestAbstractDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        assert database.session is not None

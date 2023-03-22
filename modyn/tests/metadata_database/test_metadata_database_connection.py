from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection


def get_minimal_modyn_config() -> dict:
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "host": "",
            "port": 0,
            "database": ":memory:",
        }
    }


def test_database_connection():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        assert database.session is not None


def test_register_pipeline():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        pipeline_id = database.register_pipeline(1)
        assert pipeline_id == 1
        pipeline_id = database.register_pipeline(1)
        assert pipeline_id == 2

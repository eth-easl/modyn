import pytest
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection


def get_minimal_modyn_config() -> dict:
    return {
        "storage": {
            "filesystem": {"type": "LocalFilesystemWrapper", "base_path": "/tmp/modyn"},
            "database": {
                "drivername": "sqlite",
                "username": "",
                "password": "",
                "host": "",
                "port": 0,
                "database": ":memory:",
            },
        }
    }


def get_invalid_modyn_config() -> dict:
    return {
        "storage": {
            "filesystem": {"type": "local", "base_path": "/tmp/modyn"},
            "database": {
                "drivername": "postgres",
                "username": "",
                "password": "",
                "host": "",
                "port": 10,
                "database": "/tmp/modyn/modyn.db",
            },
        }
    }


def test_database_connection():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        assert database.get_session() is not None
        assert database.add_dataset("test", "/tmp/modyn", "local", "local", "test", "0.0.1", "{}") is True


def test_get_session():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        assert database.get_session() is not None
        assert database.add_dataset("test", "/tmp/modyn", "local", "local", "test", "0.0.1", "{}") is True


def test_database_connection_with_existing_dataset():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        assert database.get_session() is not None
        assert (
            database.add_dataset(
                "test", "/tmp/modyn", "LocalFilesystemWrapper", "WebdatasetFileWrapper", "test", "0.0.1", "{}"
            )
            is True
        )
        assert (
            database.add_dataset(
                "test", "/tmp/modyn", "LocalFilesystemWrapper", "WebdatasetFileWrapper", "test", "0.0.1", "{}"
            )
            is True
        )


def test_database_connection_with_existing_dataset_and_different_base_path():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        assert database.get_session() is not None
        assert (
            database.add_dataset(
                "test", "/tmp/modyn", "LocalFilesystemWrapper", "WebdatasetFileWrapper", "test", "0.0.1", "{}"
            )
            is True
        )
        assert (
            database.add_dataset(
                "test", "/tmp/modyn2", "LocalFilesystemWrapper", "WebdatasetFileWrapper", "test", "0.0.1", "{}"
            )
            is True
        )
        assert database.get_session().query(Dataset).filter(Dataset.name == "test").first().base_path == "/tmp/modyn2"


def test_database_connection_failure():
    with pytest.raises(Exception):
        with StorageDatabaseConnection(get_invalid_modyn_config()) as database:
            database.create_tables()
            assert database.get_session() is not None
            assert (
                database.add_dataset(
                    "test", "/tmp/modyn", "LocalFilesystemWrapper", "WebdatasetFileWrapper", "test", "0.0.1", "{}"
                )
                is True
            )


def test_add_dataset_failure():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        assert (
            database.add_dataset(
                "test", "/tmp/modyn", "LocalFilesystemWrapper", "WebdatasetFileWrapper", "test", "0.0.1", "{}"
            )
            is False
        )

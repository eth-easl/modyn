import pytest
from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.database.storage_database_connection import StorageDatabaseConnection


def get_minimal_modyn_config() -> dict:
    return {
        "storage": {
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
        assert database.session is not None
        assert database.add_dataset("test", "/tmp/modyn", "local", "local", "test", "0.0.1", "{}") is True


def test_database_connection_with_existing_dataset():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        assert database.session is not None
        assert (
            database.add_dataset(
                "test", "/tmp/modyn", "LocalFilesystemWrapper", "SingleSampleFileWrapper", "test", "0.0.1", "{}"
            )
            is True
        )
        assert (
            database.add_dataset(
                "test", "/tmp/modyn", "LocalFilesystemWrapper", "SingleSampleFileWrapper", "test", "0.0.1", "{}"
            )
            is True
        )


def test_database_connection_with_existing_dataset_and_different_base_path():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        assert database.session is not None
        assert (
            database.add_dataset(
                "test", "/tmp/modyn", "LocalFilesystemWrapper", "SingleSampleFileWrapper", "test", "0.0.1", "{}"
            )
            is True
        )
        assert (
            database.add_dataset(
                "test", "/tmp/modyn2", "LocalFilesystemWrapper", "SingleSampleFileWrapper", "test", "0.0.1", "{}"
            )
            is True
        )
        assert database.session.query(Dataset).filter(Dataset.name == "test").first().base_path == "/tmp/modyn2"


def test_database_connection_failure():
    with pytest.raises(Exception):
        with StorageDatabaseConnection(get_invalid_modyn_config()) as database:
            database.create_tables()
            assert database.session is not None
            assert (
                database.add_dataset(
                    "test", "/tmp/modyn", "LocalFilesystemWrapper", "SingleSampleFileWrapper", "test", "0.0.1", "{}"
                )
                is True
            )


def test_add_dataset_failure():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        assert (
            database.add_dataset(
                "test", "/tmp/modyn", "LocalFilesystemWrapper", "SingleSampleFileWrapper", "test", "0.0.1", "{}"
            )
            is False
        )


def test_delete_dataset():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        assert database.session is not None
        assert (
            database.add_dataset(
                "test", "/tmp/modyn", "LocalFilesystemWrapper", "SingleSampleFileWrapper", "test", "0.0.1", "{}"
            )
            is True
        )
        dataset = database.session.query(Dataset).filter(Dataset.name == "test").first()
        file = File(dataset=dataset, path="/tmp/modyn/test", created_at=0, updated_at=0, number_of_samples=1)
        database.session.add(file)
        database.session.commit()
        file = database.session.query(File).filter(File.path == "/tmp/modyn/test").first()
        sample = Sample(dataset_id=dataset.dataset_id, file_id=file.file_id, index=0, label=1)
        database.session.add(sample)
        database.session.commit()
        assert database.delete_dataset("test") is True
        assert database.session.query(Dataset).filter(Dataset.name == "test").first() is None
        assert database.session.query(File).all() == []
        assert database.session.query(Sample).all() == []


def test_delete_dataset_failure():
    with StorageDatabaseConnection(get_minimal_modyn_config()) as database:
        assert database.delete_dataset("test") is False
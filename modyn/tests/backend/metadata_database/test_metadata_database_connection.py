from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.metadata_database.models.training import Training


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
        assert database.get_session() is not None


def test_get_session():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        assert database.get_session() is not None


def test_set_metadata():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        training = Training(1, 1)
        database.get_session().add(training)
        database.get_session().commit()

        database.set_metadata(["test:key:set"], [0.5], [False], [1], [b"test:data"], training.id)

        metadata = database.get_session().query(Metadata).all()
        # Because SQLite cannot handle composite primary keys with autoincrement we can't really test
        # this in a meaningful way, it is tested in the integration tests
        assert len(metadata) == 0


def test_register_training():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()

        database.get_session().query(Training).delete()

        training = database.register_training(1, 1)
        assert training == 1

        trainings = database.get_session().query(Training).all()

        assert len(trainings) == 1
        assert trainings[0].id == 1
        assert trainings[0].number_of_workers == 1
        assert trainings[0].training_set_size == 1

        training = database.register_training(2, 2)

        assert training == 2

        trainings = database.get_session().query(Training).all()
        assert len(trainings) == 2
        assert trainings[0].id == 1
        assert trainings[0].number_of_workers == 1
        assert trainings[0].training_set_size == 1
        assert trainings[1].id == 2
        assert trainings[1].number_of_workers == 2
        assert trainings[1].training_set_size == 2


def test_delete_training():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()

        database.get_session().query(Training).delete()

        training = database.register_training(1, 1)
        assert training == 1

        trainings = database.get_session().query(Training).all()

        assert len(trainings) == 1
        assert trainings[0].id == 1
        assert trainings[0].number_of_workers == 1
        assert trainings[0].training_set_size == 1

        database.delete_training(1)

        trainings = database.get_session().query(Training).all()

        assert len(trainings) == 0

        training = database.register_training(2, 2)

        assert training == 1

        trainings = database.get_session().query(Training).all()
        assert len(trainings) == 1
        assert trainings[0].id == 1
        assert trainings[0].number_of_workers == 2
        assert trainings[0].training_set_size == 2

        database.delete_training(1)

        trainings = database.get_session().query(Training).all()
        assert len(trainings) == 0


def test_get_training_information():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()

        database.get_session().query(Training).delete()

        training = Training(1, 1)

        database.get_session().add(training)
        database.get_session().commit()

        num_workers, training_set_size = database.get_training_information(training.id)
        assert num_workers == 1
        assert training_set_size == 1

        training2 = Training(2, 2)

        database.get_session().add(training2)
        database.get_session().commit()

        num_workers, training_set_size = database.get_training_information(training2.id)
        assert num_workers == 2
        assert training_set_size == 2

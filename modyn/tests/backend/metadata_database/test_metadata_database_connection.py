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
        assert database.session is not None


def test_set_metadata():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        training = Training(1)
        database.session.add(training)
        database.session.commit()

        database.set_metadata(["test:key:set"], [100], [0.5], [False], [1], [b"test:data"], training.training_id)

        metadata = database.session.query(Metadata).all()
        assert len(metadata) == 1
        assert metadata[0].key == "test:key:set"
        assert metadata[0].score == 0.5
        assert metadata[0].seen is False


def test_register_training():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()

        database.session.query(Training).delete()

        training = database.register_training(1)
        assert training == 1

        trainings = database.session.query(Training).all()

        assert len(trainings) == 1
        assert trainings[0].training_id == 1
        assert trainings[0].number_of_workers == 1

        training = database.register_training(2)

        assert training == 2

        trainings = database.session.query(Training).all()
        assert len(trainings) == 2
        assert trainings[0].training_id == 1
        assert trainings[0].number_of_workers == 1
        assert trainings[1].training_id == 2
        assert trainings[1].number_of_workers == 2


def test_delete_training():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()

        database.session.query(Training).delete()

        training = database.register_training(1)
        assert training == 1

        trainings = database.session.query(Training).all()

        assert len(trainings) == 1
        assert trainings[0].training_id == 1
        assert trainings[0].number_of_workers == 1

        database.delete_training(1)

        trainings = database.session.query(Training).all()

        assert len(trainings) == 0

        training = database.register_training(2)

        assert training == 1

        trainings = database.session.query(Training).all()
        assert len(trainings) == 1
        assert trainings[0].training_id == 1
        assert trainings[0].number_of_workers == 2

        database.delete_training(1)

        trainings = database.session.query(Training).all()
        assert len(trainings) == 0


def test_get_training_information():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()

        database.session.query(Training).delete()

        training = Training(1)

        database.session.add(training)
        database.session.commit()

        num_workers = database.get_training_information(training.training_id)
        assert num_workers == 1

        training2 = Training(2)

        database.session.add(training2)
        database.session.commit()

        num_workers = database.get_training_information(training2.training_id)
        assert num_workers == 2

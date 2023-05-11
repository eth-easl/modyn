from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import TrainedModel, Trigger


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


def test_add_trained_model():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        pipeline_id = database.register_pipeline(1)
        trigger = Trigger(pipeline_id=pipeline_id, trigger_id=5)

        database.session.add(trigger)
        database.session.commit()
        trigger_id = trigger.trigger_id

        assert pipeline_id == 1 and trigger_id == 5

        model_id = database.add_trained_model(pipeline_id, trigger_id, "test_path.modyn")

        model: TrainedModel = database.session.get(TrainedModel, model_id)

        assert model.model_id == 1
        assert model.model_path == "test_path.modyn"
        assert model.pipeline_id == 1 and model.trigger_id == 5

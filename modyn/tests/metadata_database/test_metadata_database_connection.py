import json

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import TrainedModel, Trigger
from modyn.metadata_database.utils import ModelStorageStrategyConfig


def get_minimal_modyn_config() -> dict:
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "hostname": "",
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
        pipeline_id = database.register_pipeline(
            1, "ResNet18", "{}", True, "{}", "{}", ModelStorageStrategyConfig(name="PyTorchFullModel")
        )
        assert pipeline_id == 1
        pipeline_id = database.register_pipeline(
            1, "ResNet18", "{}", False, "{}", "{}", ModelStorageStrategyConfig(name="PyTorchFullModel")
        )
        assert pipeline_id == 2


def test_add_trained_model():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        pipeline_id = database.register_pipeline(
            1, "ResNet18", "{}", True, "{}", "{}", ModelStorageStrategyConfig(name="PyTorchFullModel")
        )
        trigger = Trigger(pipeline_id=pipeline_id, trigger_id=5)

        database.session.add(trigger)
        database.session.commit()
        trigger_id = trigger.trigger_id

        assert pipeline_id == 1 and trigger_id == 5

        model_id = database.add_trained_model(pipeline_id, trigger_id, "test_path.modyn", "test_path.metadata")

        model_parent: TrainedModel = database.session.get(TrainedModel, model_id)

        assert model_parent.model_id == 1
        assert model_parent.model_path == "test_path.modyn"
        assert model_parent.metadata_path == "test_path.metadata"
        assert model_parent.pipeline_id == 1 and model_parent.trigger_id == 5
        assert model_parent.parent_model is None

        model_id = database.add_trained_model(
            pipeline_id, 6, "test_path.modyn", "test_path.metadata", parent_model=model_parent.model_id
        )
        model_child: TrainedModel = database.session.get(TrainedModel, model_id)

        assert model_child.parent_model == model_parent.model_id
        assert len(model_parent.children) == 1
        assert model_parent.children[0] == model_child


def test_get_model_configuration():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
        pipeline_id = database.register_pipeline(
            1,
            "ResNet18",
            json.dumps({"num_classes": 10}),
            True,
            "{}",
            "{}",
            ModelStorageStrategyConfig(name="PyTorchFullModel"),
        )

        assert pipeline_id == 1

        model_class_name, model_config, amp = database.get_model_configuration(pipeline_id)

        assert model_class_name == "ResNet18"
        assert json.loads(model_config) == {"num_classes": 10}
        assert amp

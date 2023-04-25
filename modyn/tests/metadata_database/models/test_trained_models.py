# pylint: disable=redefined-outer-name
import pytest
from modyn.metadata_database.models import TrainedModel, Trigger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    Trigger.metadata.create_all(engine)
    TrainedModel.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_trained_model(session):
    model = TrainedModel(pipeline_id=1, trigger_id=1, model_path="test_path")
    session.add(model)
    session.commit()

    assert session.query(TrainedModel).filter(TrainedModel.trigger_id == 1).first() is not None
    assert session.query(TrainedModel).filter(TrainedModel.trigger_id == 1).first().pipeline_id == 1


def test_get_trained_model(session):
    model = TrainedModel(pipeline_id=1, trigger_id=1, model_path="test_path")
    session.add(model)
    session.commit()

    fetched_valid = session.get(TrainedModel, 1)
    fetched_invalid = session.get(TrainedModel, 2)

    assert fetched_valid.model_id == 1
    assert fetched_valid.model_path == "test_path"
    assert fetched_invalid is None

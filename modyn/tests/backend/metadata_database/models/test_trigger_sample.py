# pylint: disable=redefined-outer-name
import pytest
from modyn.backend.metadata_database.models import TriggerSample
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    TriggerSample.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_trigger_sample_list(session):
    trigger_sample_list = TriggerSample(trigger_id=1, pipeline_id=1, partition_id=1, sample_key=0, sample_weight=1.0)
    session.add(trigger_sample_list)
    session.commit()

    assert session.query(TriggerSample).filter(TriggerSample.trigger_sample_list_id == 1).first() is not None
    assert session.query(TriggerSample).filter(TriggerSample.trigger_id == 1).first().trigger_sample_list_id == 1
    assert session.query(TriggerSample).filter(TriggerSample.pipeline_id == 1).first().sample_key == 0
    assert session.query(TriggerSample).filter(TriggerSample.pipeline_id == 1).first().partition_id == 1


def test_update_trigger_sample_list(session):
    trigger_sample_list = TriggerSample(trigger_id=1, pipeline_id=1, partition_id=1, sample_key=0, sample_weight=1.0)
    session.add(trigger_sample_list)
    session.commit()

    trigger_sample_list.seen_by_trigger = False
    trigger_sample_list.part_of_training_set = False
    session.commit()

    assert session.query(TriggerSample).filter(TriggerSample.trigger_sample_list_id == 1).first() is not None
    assert session.query(TriggerSample).filter(TriggerSample.trigger_sample_list_id == 1).first().trigger_id == 1
    assert session.query(TriggerSample).filter(TriggerSample.trigger_sample_list_id == 1).first().partition_id == 1

    assert session.query(TriggerSample).filter(TriggerSample.trigger_sample_list_id == 1).first().sample_key == 0


def test_delete_trigger_sample_list(session):
    trigger_sample_list = TriggerSample(trigger_id=1, pipeline_id=1, partition_id=1, sample_key=0, sample_weight=1.0)
    session.add(trigger_sample_list)
    session.commit()

    session.delete(trigger_sample_list)
    session.commit()

    assert session.query(TriggerSample).filter(TriggerSample.trigger_sample_list_id == 1).first() is None

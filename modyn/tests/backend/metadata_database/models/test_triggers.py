# pylint: disable=redefined-outer-name
import pytest
from modyn.backend.metadata_database.models import Pipeline, Trigger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    Trigger.metadata.create_all(engine)
    Pipeline.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_trigger(session):
    pipeline = Pipeline(
        num_workers=42,
    )
    session.add(pipeline)
    session.commit()
    trigger = Trigger(
        trigger_id=1,
        pipeline=pipeline,
    )
    session.add(trigger)
    session.commit()

    assert session.query(Trigger).filter(Trigger.trigger_id == 1).first() is not None
    assert session.query(Trigger).filter(Trigger.trigger_id == 1).first().pipeline_id == 1
    assert session.query(Trigger).filter(Trigger.trigger_id == 1).first().pipeline.num_workers == 42


def test_update_trigger(session):
    pipeline = Pipeline(
        num_workers=42,
    )
    session.add(pipeline)
    session.commit()
    trigger = Trigger(
        trigger_id=1,
        pipeline=pipeline,
    )
    session.add(trigger)
    session.commit()

    trigger.pipeline_id = 1
    session.commit()

    assert session.query(Trigger).filter(Trigger.pipeline_id == 1).first() is not None


def test_delete_trigger(session):
    pipeline = Pipeline(
        num_workers=42,
    )
    session.add(pipeline)
    session.commit()
    trigger = Trigger(
        trigger_id=1,
        pipeline=pipeline,
    )
    session.add(trigger)
    session.commit()

    session.delete(trigger)
    session.commit()

    assert session.query(Trigger).filter(Trigger.trigger_id == 1).first() is None


def test_repr(session):
    pipeline = Pipeline(
        num_workers=42,
    )
    session.add(pipeline)
    session.commit()
    trigger = Trigger(
        trigger_id=1,
        pipeline=pipeline,
    )
    session.add(trigger)
    session.commit()

    assert repr(trigger) == "<Trigger 1:1>"

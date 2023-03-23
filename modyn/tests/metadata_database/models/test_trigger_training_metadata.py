# pylint: disable=redefined-outer-name
import pytest
from modyn.metadata_database.models import TriggerTrainingMetadata
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    TriggerTrainingMetadata.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_trigger_training_metadata(session):
    trigger_training_metadata = TriggerTrainingMetadata(
        trigger_id=1,
        pipeline_id=1,
        time_to_train=1.0,
        overall_loss=1.0,
    )

    session.add(trigger_training_metadata)
    session.commit()

    assert (
        session.query(TriggerTrainingMetadata).filter(TriggerTrainingMetadata.trigger_training_metadata_id == 1).first()
        is not None
    )
    assert (
        session.query(TriggerTrainingMetadata)
        .filter(TriggerTrainingMetadata.trigger_training_metadata_id == 1)
        .first()
        .trigger_id
        == 1
    )
    assert (
        session.query(TriggerTrainingMetadata)
        .filter(TriggerTrainingMetadata.trigger_training_metadata_id == 1)
        .first()
        .time_to_train
        == 1.0
    )
    assert (
        session.query(TriggerTrainingMetadata)
        .filter(TriggerTrainingMetadata.trigger_training_metadata_id == 1)
        .first()
        .overall_loss
        == 1.0
    )


def test_update_trigger_training_metadata(session):
    trigger_training_metadata = TriggerTrainingMetadata(
        trigger_id=1,
        pipeline_id=1,
        time_to_train=1.0,
        overall_loss=1.0,
    )

    session.add(trigger_training_metadata)
    session.commit()

    trigger_training_metadata.time_to_train = 2.0
    trigger_training_metadata.overall_loss = 2.0
    session.commit()

    assert (
        session.query(TriggerTrainingMetadata).filter(TriggerTrainingMetadata.trigger_training_metadata_id == 1).first()
        is not None
    )
    assert (
        session.query(TriggerTrainingMetadata)
        .filter(TriggerTrainingMetadata.trigger_training_metadata_id == 1)
        .first()
        .trigger_id
        == 1
    )
    assert (
        session.query(TriggerTrainingMetadata)
        .filter(TriggerTrainingMetadata.trigger_training_metadata_id == 1)
        .first()
        .time_to_train
        == 2.0
    )
    assert (
        session.query(TriggerTrainingMetadata)
        .filter(TriggerTrainingMetadata.trigger_training_metadata_id == 1)
        .first()
        .overall_loss
        == 2.0
    )


def test_delete_trigger_training_metadata(session):
    trigger_training_metadata = TriggerTrainingMetadata(
        trigger_id=1,
        pipeline_id=1,
        time_to_train=1.0,
        overall_loss=1.0,
    )

    session.add(trigger_training_metadata)
    session.commit()

    session.delete(trigger_training_metadata)
    session.commit()

    assert (
        session.query(TriggerTrainingMetadata).filter(TriggerTrainingMetadata.trigger_training_metadata_id == 1).first()
        is None
    )

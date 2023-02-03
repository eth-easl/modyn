# pylint: disable=redefined-outer-name
import pytest
from modyn.backend.metadata_database.models.trigger_sample_training_metadata import TriggerSampleTrainingMetadata
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    TriggerSampleTrainingMetadata.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_trigger_sample_training_metadata(session):
    trigger_sample_training_metadata = TriggerSampleTrainingMetadata(
        trigger_training_metadata_id=1,
        sample_id="sample_id",
        seen_by_trigger=True,
        part_of_training_set=True,
    )
    session.add(trigger_sample_training_metadata)
    session.commit()

    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        is not None
    )
    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        .trigger_training_metadata_id
        == 1
    )
    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        .sample_id
        == "sample_id"
    )
    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        .seen_by_trigger
        is True
    )
    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        .part_of_training_set
        is True
    )


def test_update_trigger_sample_training_metadata(session):
    trigger_sample_training_metadata = TriggerSampleTrainingMetadata(
        trigger_training_metadata_id=1,
        sample_id="sample_id",
        seen_by_trigger=True,
        part_of_training_set=True,
    )
    session.add(trigger_sample_training_metadata)
    session.commit()

    trigger_sample_training_metadata.seen_by_trigger = False
    trigger_sample_training_metadata.part_of_training_set = False
    session.commit()

    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        is not None
    )
    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        .trigger_training_metadata_id
        == 1
    )
    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        .sample_id
        == "sample_id"
    )
    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        .seen_by_trigger
        is False
    )
    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        .part_of_training_set
        is False
    )


def test_delete_trigger_sample_training_metadata(session):
    trigger_sample_training_metadata = TriggerSampleTrainingMetadata(
        trigger_training_metadata_id=1,
        sample_id="sample_id",
        seen_by_trigger=True,
        part_of_training_set=True,
    )
    session.add(trigger_sample_training_metadata)
    session.commit()

    session.delete(trigger_sample_training_metadata)
    session.commit()

    assert (
        session.query(TriggerSampleTrainingMetadata)
        .filter(TriggerSampleTrainingMetadata.trigger_sample_training_metadata_id == 1)
        .first()
        is None
    )


def test_repr(session):
    trigger_sample_training_metadata = TriggerSampleTrainingMetadata(
        trigger_training_metadata_id=1,
        sample_id="sample_id",
        seen_by_trigger=True,
        part_of_training_set=True,
    )
    session.add(trigger_sample_training_metadata)
    session.commit()

    assert repr(trigger_sample_training_metadata) == "<TriggerSampleTrainingMetadata 1:sample_id>"

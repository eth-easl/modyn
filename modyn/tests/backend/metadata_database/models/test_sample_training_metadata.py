# pylint: disable=redefined-outer-name
import pytest
from modyn.backend.metadata_database.models import SampleTrainingMetadata
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    SampleTrainingMetadata.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_sample_training_metadata(session):
    sample_training_metadata = SampleTrainingMetadata(
        pipeline_id=1,
        trigger_id=1,
        sample_key="sample_key",
        loss=1.0,
        gradient=1.0,
    )
    session.add(sample_training_metadata)
    session.commit()

    assert (
        session.query(SampleTrainingMetadata).filter(SampleTrainingMetadata.sample_training_metadata_id == 1).first()
        is not None
    )
    assert (
        session.query(SampleTrainingMetadata)
        .filter(SampleTrainingMetadata.sample_training_metadata_id == 1)
        .first()
        .pipeline_id
        == 1
    )
    assert (
        session.query(SampleTrainingMetadata)
        .filter(SampleTrainingMetadata.sample_training_metadata_id == 1)
        .first()
        .trigger_id
        == 1
    )
    assert (
        session.query(SampleTrainingMetadata)
        .filter(SampleTrainingMetadata.sample_training_metadata_id == 1)
        .first()
        .sample_key
        == "sample_key"
    )
    assert (
        session.query(SampleTrainingMetadata)
        .filter(SampleTrainingMetadata.sample_training_metadata_id == 1)
        .first()
        .loss
        == 1.0
    )
    assert (
        session.query(SampleTrainingMetadata)
        .filter(SampleTrainingMetadata.sample_training_metadata_id == 1)
        .first()
        .gradient
        == 1.0
    )


def test_update_sample_training_metadata(session):
    sample_training_metadata = SampleTrainingMetadata(
        pipeline_id=1,
        trigger_id=1,
        sample_key="sample_key",
        loss=1.0,
        gradient=1.0,
    )
    session.add(sample_training_metadata)
    session.commit()

    sample_training_metadata.loss = 2.0
    sample_training_metadata.gradient = 2.0
    session.commit()

    assert (
        session.query(SampleTrainingMetadata).filter(SampleTrainingMetadata.sample_training_metadata_id == 1).first()
        is not None
    )
    assert (
        session.query(SampleTrainingMetadata)
        .filter(SampleTrainingMetadata.sample_training_metadata_id == 1)
        .first()
        .pipeline_id
        == 1
    )
    assert (
        session.query(SampleTrainingMetadata)
        .filter(SampleTrainingMetadata.sample_training_metadata_id == 1)
        .first()
        .trigger_id
        == 1
    )
    assert (
        session.query(SampleTrainingMetadata)
        .filter(SampleTrainingMetadata.sample_training_metadata_id == 1)
        .first()
        .sample_key
        == "sample_key"
    )
    assert (
        session.query(SampleTrainingMetadata)
        .filter(SampleTrainingMetadata.sample_training_metadata_id == 1)
        .first()
        .loss
        == 2.0
    )
    assert (
        session.query(SampleTrainingMetadata)
        .filter(SampleTrainingMetadata.sample_training_metadata_id == 1)
        .first()
        .gradient
        == 2.0
    )


def test_delete_sample_training_metadata(session):
    sample_training_metadata = SampleTrainingMetadata(
        pipeline_id=1,
        trigger_id=1,
        sample_key="sample_key",
        loss=1.0,
        gradient=1.0,
    )
    session.add(sample_training_metadata)
    session.commit()

    session.delete(sample_training_metadata)
    session.commit()

    assert (
        session.query(SampleTrainingMetadata).filter(SampleTrainingMetadata.sample_training_metadata_id == 1).first()
        is None
    )


def test_repr(session):
    sample_training_metadata = SampleTrainingMetadata(
        pipeline_id=1,
        trigger_id=1,
        sample_key="sample_key",
        loss=1.0,
        gradient=1.0,
    )
    session.add(sample_training_metadata)
    session.commit()

    assert repr(sample_training_metadata) == "<SampleTrainingMetadata 1:1:sample_key>"

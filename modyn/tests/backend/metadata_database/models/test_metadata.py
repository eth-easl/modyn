# pylint: disable=redefined-outer-name
import pytest
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.metadata_database.models.training import Training
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# TODO(#113): there is no training ID anymore


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    Metadata.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_metadata(session):
    training = Training(number_of_workers=1)

    session.add(training)
    session.commit()

    metadata = Metadata(
        key="test_key",
        timestamp=100,
        score=0.5,
        seen=False,
        label=1,
        data=b"test_data",
        pipeline_id=training.training_id,
        trigger_id=42,
    )

    metadata.metadata_id = 1

    session.add(metadata)
    session.commit()

    assert session.query(Metadata).filter(Metadata.key == "test_key").first() is not None
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().key == "test_key"
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().score == 0.5
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().seen is False
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().pipeline_id == 1
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().data == b"test_data"
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().trigger_id == 42


def test_update_metadata(session):
    training = Training(number_of_workers=1)

    session.add(training)
    session.commit()

    metadata = Metadata(
        key="test_key",
        timestamp=100,
        score=0.5,
        seen=False,
        label=1,
        data=b"test_data",
        pipeline_id=training.training_id,
        trigger_id=42,
    )

    metadata.metadata_id = 1  # This is because SQLite does not support autoincrement for composite primary keys

    session.add(metadata)
    session.commit()

    metadata.value = 0.6
    metadata.is_json = True
    metadata.data = b"test_data_2"

    session.commit()

    assert session.query(Metadata).filter(Metadata.key == "test_key").first() is not None
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().key == "test_key"
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().value == 0.6
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().is_json is True
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().pipeline_id == 1
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().data == b"test_data_2"
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().trigger_id == 42


def test_delete_metadata(session):
    training = Training(number_of_workers=1)

    session.add(training)
    session.commit()

    metadata = Metadata(
        key="test_key",
        timestamp=100,
        score=0.5,
        seen=False,
        label=1,
        data=b"test_data",
        pipeline_id=training.training_id,
        trigger_id=42,
    )

    metadata.metadata_id = 1  # This is because SQLite does not support autoincrement for composite primary keys

    session.add(metadata)
    session.commit()

    session.delete(metadata)
    session.commit()

    assert session.query(Metadata).filter(Metadata.key == "test_key").first() is None


def test_repr_metadata(session):
    training = Training(number_of_workers=1)

    session.add(training)
    session.commit()

    metadata = Metadata(
        key="test_key",
        timestamp=100,
        score=0.5,
        seen=False,
        label=1,
        data=b"test_data",
        pipeline_id=training.training_id,
        trigger_id=42,
    )

    metadata.metadata_id = 1  # This is because SQLite does not support autoincrement for composite primary keys

    session.add(metadata)
    session.commit()

    assert repr(metadata) == "<Metadata test_key>"

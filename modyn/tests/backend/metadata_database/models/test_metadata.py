# pylint: disable=redefined-outer-name
import pytest
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.metadata_database.models.training import Training
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    Metadata.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_metadata(session):
    training = Training(1, 1)

    session.add(training)
    session.commit()

    metadata = Metadata(
        "test_key",
        0.5,
        False,
        1,
        b"test_data",
        training.id,
    )

    metadata.id = 1

    session.add(metadata)
    session.commit()

    assert session.query(Metadata).filter(Metadata.key == "test_key").first() is not None
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().key == "test_key"
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().score == 0.5
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().seen is False
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().training_id == 1
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().data == b"test_data"


def test_update_metadata(session):
    training = Training(1, 1)

    session.add(training)
    session.commit()

    metadata = Metadata(
        "test_key",
        0.5,
        False,
        1,
        b"test_data",
        training.id,
    )

    metadata.id = 1  # This is because SQLite does not support autoincrement for composite primary keys

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
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().training_id == 1
    assert session.query(Metadata).filter(Metadata.key == "test_key").first().data == b"test_data_2"


def test_delete_metadata(session):
    training = Training(1, 1)

    session.add(training)
    session.commit()

    metadata = Metadata(
        "test_key",
        0.5,
        False,
        1,
        b"test_data",
        training.id,
    )

    metadata.id = 1  # This is because SQLite does not support autoincrement for composite primary keys

    session.add(metadata)
    session.commit()

    session.delete(metadata)
    session.commit()

    assert session.query(Metadata).filter(Metadata.key == "test_key").first() is None


def test_repr_metadata(session):
    training = Training(1, 1)

    session.add(training)
    session.commit()

    metadata = Metadata(
        "test_key",
        0.5,
        False,
        1,
        b"test_data",
        training.id,
    )

    metadata.id = 1  # This is because SQLite does not support autoincrement for composite primary keys

    session.add(metadata)
    session.commit()

    assert repr(metadata) == "<Metadata test_key>"

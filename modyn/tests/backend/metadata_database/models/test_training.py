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


def test_add_training(session):
    training = Training(1, 1)

    session.add(training)
    session.commit()

    assert session.query(Training).filter(Training.training_id == 1).first() is not None
    assert session.query(Training).filter(Training.training_id == 1).first().training_id == 1
    assert session.query(Training).filter(Training.training_id == 1).first().number_of_workers == 1
    assert session.query(Training).filter(Training.training_id == 1).first().training_set_size == 1


def test_update_training(session):
    training = Training(1, 1)

    session.add(training)
    session.commit()

    training = session.query(Training).filter(Training.training_id == 1).first()
    training.number_of_workers = 2
    training.training_set_size = 2

    session.commit()

    assert session.query(Training).filter(Training.training_id == 1).first() is not None
    assert session.query(Training).filter(Training.training_id == 1).first().training_id == 1
    assert session.query(Training).filter(Training.training_id == 1).first().number_of_workers == 2
    assert session.query(Training).filter(Training.training_id == 1).first().training_set_size == 2


def test_delete_training(session):
    training = Training(1, 1)

    session.add(training)
    session.commit()

    session.delete(training)
    session.commit()

    assert session.query(Training).filter(Training.training_id == 1).first() is None


def test_repr(session):
    training = Training(1, 1)

    session.add(training)
    session.commit()

    assert repr(training) == "<Training 1>"

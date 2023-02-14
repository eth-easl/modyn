# pylint: disable=redefined-outer-name
import pytest
from modyn.backend.metadata_database.models import Pipeline
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    Pipeline.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_pipeline(session):
    pipeline = Pipeline(
        num_workers=10,
    )
    session.add(pipeline)
    session.commit()

    assert session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first() is not None
    assert session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first().num_workers == 10


def test_update_pipeline(session):
    pipeline = Pipeline(
        num_workers=10,
    )
    session.add(pipeline)
    session.commit()

    pipeline.num_workers = 20
    session.commit()

    assert session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first() is not None
    assert session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first().num_workers == 20


def test_delete_pipeline(session):
    pipeline = Pipeline(
        num_workers=10,
    )
    session.add(pipeline)
    session.commit()

    session.delete(pipeline)
    session.commit()

    assert session.query(Pipeline).filter(Pipeline.pipeline_id == 1).first() is None

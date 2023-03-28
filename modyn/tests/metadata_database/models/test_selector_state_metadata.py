# pylint: disable=redefined-outer-name
import pytest
from modyn.metadata_database.models import SelectorStateMetadata
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    SelectorStateMetadata.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_selector_state_metadata(session):
    selector_state_metadata = SelectorStateMetadata(
        pipeline_id=1,
        sample_key=0,
        seen_in_trigger_id=1,
        used=False,
        timestamp=10000,
        label=10,
    )
    session.add(selector_state_metadata)
    session.commit()

    assert (
        session.query(SelectorStateMetadata).filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1).first()
        is not None
    )
    assert (
        session.query(SelectorStateMetadata)
        .filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1)
        .first()
        .pipeline_id
        == 1
    )
    assert (
        session.query(SelectorStateMetadata)
        .filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1)
        .first()
        .sample_key
        == 0
    )
    assert (
        session.query(SelectorStateMetadata)
        .filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1)
        .first()
        .seen_in_trigger_id
        == 1
    )
    assert (
        session.query(SelectorStateMetadata).filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1).first().used
        is False
    )
    assert (
        session.query(SelectorStateMetadata)
        .filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1)
        .first()
        .timestamp
        == 10000
    )
    assert (
        session.query(SelectorStateMetadata).filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1).first().label
        == 10
    )


def test_update_selector_state_metadata(session):
    selector_state_metadata = SelectorStateMetadata(
        pipeline_id=1,
        sample_key=0,
        seen_in_trigger_id=1,
        used=False,
        timestamp=10000,
        label=10,
    )
    session.add(selector_state_metadata)
    session.commit()

    selector_state_metadata.score = 2.0
    selector_state_metadata.timestamp = 20000
    selector_state_metadata.label = 20
    session.commit()

    assert (
        session.query(SelectorStateMetadata).filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1).first()
        is not None
    )
    assert (
        session.query(SelectorStateMetadata)
        .filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1)
        .first()
        .pipeline_id
        == 1
    )
    assert (
        session.query(SelectorStateMetadata)
        .filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1)
        .first()
        .sample_key
        == 0
    )
    assert (
        session.query(SelectorStateMetadata)
        .filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1)
        .first()
        .seen_in_trigger_id
        == 1
    )
    assert (
        session.query(SelectorStateMetadata).filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1).first().used
        is False
    )
    assert (
        session.query(SelectorStateMetadata).filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1).first().score
        == 2.0
    )
    assert (
        session.query(SelectorStateMetadata)
        .filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1)
        .first()
        .timestamp
        == 20000
    )
    assert (
        session.query(SelectorStateMetadata).filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1).first().label
        == 20
    )


def test_delete_selector_state_metadata(session):
    selector_state_metadata = SelectorStateMetadata(
        pipeline_id=1,
        sample_key=0,
        seen_in_trigger_id=1,
        used=False,
        timestamp=10000,
        label=10,
    )
    session.add(selector_state_metadata)
    session.commit()

    session.delete(selector_state_metadata)
    session.commit()

    assert (
        session.query(SelectorStateMetadata).filter(SelectorStateMetadata.pipeline_id == 1, SelectorStateMetadata.sample_key == 0, SelectorStateMetadata.seen_in_trigger_id == 1).first()
        is None
    )

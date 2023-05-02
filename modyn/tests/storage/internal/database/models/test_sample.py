# pylint: disable=redefined-outer-name
import pytest
from modyn.storage.internal.database.models import Dataset, File, Sample
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType
from modyn.utils import current_time_millis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

NOW = current_time_millis()


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()
    Sample.ensure_pks_correct(sess)

    Dataset.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_sample(session):
    dataset = Dataset(
        name="test",
        base_path="test",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        description="test",
        version="test",
        last_timestamp=0,
    )
    session.add(dataset)
    session.commit()

    now = NOW
    file = File(dataset=dataset, path="test", created_at=now, updated_at=now, number_of_samples=0)
    session.add(file)
    session.commit()

    sample = Sample(dataset_id=dataset.dataset_id, file_id=file.file_id, index=0, label=b"test")
    session.add(sample)
    session.commit()

    sample_id = sample.sample_id

    assert session.query(Sample).filter(Sample.sample_id == sample_id).first() is not None
    assert session.query(Sample).filter(Sample.sample_id == sample_id).first().file_id == file.file_id
    assert session.query(Sample).filter(Sample.sample_id == sample_id).first().index == 0
    assert session.query(Sample).filter(Sample.sample_id == sample_id).first().label == b"test"


def test_update_sample(session):
    dataset = Dataset(
        name="test",
        base_path="test",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        description="test",
        version="test",
        last_timestamp=0,
    )
    session.add(dataset)
    session.commit()

    now = NOW
    file = File(dataset=dataset, path="test", created_at=now, updated_at=now, number_of_samples=0)
    session.add(file)
    session.commit()

    sample = Sample(dataset_id=dataset.dataset_id, file_id=file.file_id, index=0, label=b"test")
    session.add(sample)
    session.commit()

    sample_id = sample.sample_id

    session.query(Sample).filter(Sample.sample_id == sample_id).update({"index": 1})

    assert session.query(Sample).filter(Sample.sample_id == sample_id).first().index == 1


def test_delete_sample(session):
    dataset = Dataset(
        name="test",
        base_path="test",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        description="test",
        version="test",
        last_timestamp=0,
    )
    session.add(dataset)
    session.commit()

    now = NOW
    file = File(dataset=dataset, path="test", created_at=now, updated_at=now, number_of_samples=0)
    session.add(file)
    session.commit()

    sample = Sample(dataset_id=dataset.dataset_id, file_id=file.file_id, index=0, label=b"test")
    session.add(sample)
    session.commit()

    sample_id = sample.sample_id

    session.query(Sample).filter(Sample.sample_id == sample_id).delete()

    assert session.query(Sample).filter(Sample.sample_id == sample_id).first() is None


def test_repr(session):
    dataset = Dataset(
        name="test",
        base_path="test",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        description="test",
        version="test",
        last_timestamp=0,
    )
    session.add(dataset)
    session.commit()

    now = NOW
    file = File(dataset=dataset, path="test", created_at=now, updated_at=now, number_of_samples=0)
    session.add(file)
    session.commit()

    sample = Sample(dataset_id=dataset.dataset_id, file_id=file.file_id, index=0, label=b"test")
    session.add(sample)
    session.commit()

    assert repr(sample) == "<Sample 1>"

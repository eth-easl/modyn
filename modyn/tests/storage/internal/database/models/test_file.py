# pylint: disable=redefined-outer-name
import pytest
from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.database.models.file import File
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

    Dataset.metadata.create_all(engine)
    File.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_file(session):
    dataset = Dataset(
        name="test",
        base_path="test",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        description="test",
        version="test",
    )
    session.add(dataset)
    session.commit()

    now = NOW
    file = File(dataset=dataset, path="test", created_at=now, updated_at=now, number_of_samples=0)
    session.add(file)
    session.commit()

    assert session.query(File).filter(File.path == "test").first() is not None
    assert session.query(File).filter(File.path == "test").first().dataset == dataset
    assert session.query(File).filter(File.path == "test").first().created_at == now
    assert session.query(File).filter(File.path == "test").first().updated_at == now


def test_update_file(session):
    dataset = Dataset(
        name="test",
        base_path="test",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        description="test",
        version="test",
    )
    session.add(dataset)
    session.commit()

    now = NOW
    file = File(dataset=dataset, path="test", created_at=now, updated_at=now, number_of_samples=0)
    session.add(file)
    session.commit()

    now = NOW

    session.query(File).filter(File.path == "test").update({"path": "test2", "created_at": now, "updated_at": now})
    session.commit()

    assert session.query(File).filter(File.path == "test2").first() is not None
    assert session.query(File).filter(File.path == "test2").first().dataset == dataset
    assert session.query(File).filter(File.path == "test2").first().created_at == now
    assert session.query(File).filter(File.path == "test2").first().updated_at == now


def test_delete_file(session):
    dataset = Dataset(
        name="test",
        base_path="test",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        description="test",
        version="test",
    )
    session.add(dataset)
    session.commit()

    now = NOW
    file = File(dataset=dataset, path="test", created_at=now, updated_at=now, number_of_samples=0)
    session.add(file)
    session.commit()

    session.query(File).filter(File.path == "test").delete()
    session.commit()

    assert session.query(File).filter(File.path == "test").first() is None


def test_repr_file(session):
    dataset = Dataset(
        name="test",
        base_path="test",
        filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
        file_wrapper_type=FileWrapperType.SingleSampleFileWrapper,
        description="test",
        version="test",
    )
    session.add(dataset)
    session.commit()

    now = NOW
    file = File(dataset=dataset, path="test", created_at=now, updated_at=now, number_of_samples=0)
    session.add(file)
    session.commit()

    assert repr(file) == "<File test>"

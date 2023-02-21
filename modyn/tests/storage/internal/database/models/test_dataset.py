# pylint: disable=redefined-outer-name
import pytest
from modyn.storage.internal.database.models import Dataset
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def session():
    engine = create_engine("sqlite:///:memory:", echo=True)
    sess = sessionmaker(bind=engine)()

    Dataset.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_dataset(session):
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

    assert session.query(Dataset).filter(Dataset.name == "test").first() is not None
    assert session.query(Dataset).filter(Dataset.name == "test").first().base_path == "test"
    assert (
        session.query(Dataset).filter(Dataset.name == "test").first().filesystem_wrapper_type
        == FilesystemWrapperType.LocalFilesystemWrapper
    )
    assert (
        session.query(Dataset).filter(Dataset.name == "test").first().file_wrapper_type
        == FileWrapperType.SingleSampleFileWrapper
    )
    assert session.query(Dataset).filter(Dataset.name == "test").first().description == "test"
    assert session.query(Dataset).filter(Dataset.name == "test").first().version == "test"


def test_update_dataset(session):
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

    session.query(Dataset).filter(Dataset.name == "test").update(
        {
            "base_path": "test2",
            "file_wrapper_type": FileWrapperType.SingleSampleFileWrapper,
            "description": "test2",
            "version": "test2",
        }
    )
    session.commit()

    assert session.query(Dataset).filter(Dataset.name == "test").first().base_path == "test2"
    assert (
        session.query(Dataset).filter(Dataset.name == "test").first().file_wrapper_type
        == FileWrapperType.SingleSampleFileWrapper
    )
    assert session.query(Dataset).filter(Dataset.name == "test").first().description == "test2"
    assert session.query(Dataset).filter(Dataset.name == "test").first().version == "test2"


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

    assert repr(dataset) == "<Dataset test>"


def test_delete_dataset(session):
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

    session.query(Dataset).filter(Dataset.name == "test").delete()
    session.commit()

    assert session.query(Dataset).filter(Dataset.name == "test").first() is None

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modyn.storage.internal.database.models.dataset import Dataset
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType


@pytest.fixture(autouse=True)
def session():
    engine = create_engine('sqlite:///:memory:', echo=True)
    sess = sessionmaker(bind=engine)()

    Dataset.metadata.create_all(engine)

    yield sess

    sess.close()
    engine.dispose()


def test_add_dataset(session):  # pylint: disable=redefined-outer-name
    dataset = Dataset(name='test',
                      base_path='test',
                      filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
                      file_wrapper_type=FileWrapperType.MNISTWebdatasetFileWrapper,
                      description='test',
                      version='test')
    session.add(dataset)
    session.commit()

    assert session.query(Dataset).filter(Dataset.name == 'test').first() is not None
    assert session.query(Dataset).filter(Dataset.name == 'test').first().base_path == 'test'
    assert session.query(Dataset).filter(Dataset.name == 'test') \
                  .first().filesystem_wrapper_type == FilesystemWrapperType.LocalFilesystemWrapper
    assert session.query(Dataset).filter(Dataset.name == 'test') \
                  .first().file_wrapper_type == FileWrapperType.MNISTWebdatasetFileWrapper
    assert session.query(Dataset).filter(Dataset.name == 'test').first().description == 'test'
    assert session.query(Dataset).filter(Dataset.name == 'test').first().version == 'test'


def test_update_dataset(session):  # pylint: disable=redefined-outer-name
    dataset = Dataset(name='test',
                      base_path='test',
                      filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
                      file_wrapper_type=FileWrapperType.MNISTWebdatasetFileWrapper,
                      description='test',
                      version='test')
    session.add(dataset)
    session.commit()

    session.query(Dataset).filter(Dataset.name == 'test').update({
        'base_path': 'test2',
        'filesystem_wrapper_type': FilesystemWrapperType.S3FileSystemWrapper,
        'file_wrapper_type': FileWrapperType.ParquetFileWrapper,
        'description': 'test2',
        'version': 'test2'
    })
    session.commit()

    assert session.query(Dataset).filter(Dataset.name == 'test') \
                  .first().base_path == 'test2'
    assert session.query(Dataset).filter(Dataset.name == 'test') \
                  .first().filesystem_wrapper_type == FilesystemWrapperType.S3FileSystemWrapper
    assert session.query(Dataset).filter(Dataset.name == 'test') \
                  .first().file_wrapper_type == FileWrapperType.ParquetFileWrapper
    assert session.query(Dataset).filter(Dataset.name == 'test').first().description == 'test2'
    assert session.query(Dataset).filter(Dataset.name == 'test').first().version == 'test2'


def test_repr(session):  # pylint: disable=redefined-outer-name
    dataset = Dataset(name='test',
                      base_path='test',
                      filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
                      file_wrapper_type=FileWrapperType.MNISTWebdatasetFileWrapper,
                      description='test',
                      version='test')
    session.add(dataset)
    session.commit()

    assert repr(dataset) == '<Dataset test>'


def test_delete_dataset(session):  # pylint: disable=redefined-outer-name
    dataset = Dataset(name='test',
                      base_path='test',
                      filesystem_wrapper_type=FilesystemWrapperType.LocalFilesystemWrapper,
                      file_wrapper_type=FileWrapperType.MNISTWebdatasetFileWrapper,
                      description='test',
                      version='test')
    session.add(dataset)
    session.commit()

    session.query(Dataset).filter(Dataset.name == 'test').delete()
    session.commit()

    assert session.query(Dataset).filter(Dataset.name == 'test').first() is None

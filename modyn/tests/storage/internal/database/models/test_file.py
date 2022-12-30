import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.dataset import Dataset


SESSION = None
ENGINE = None


def setup():
    global ENGINE, SESSION  # pylint: disable=global-statement # noqa: E262
    ENGINE = create_engine('sqlite:///:memory:', echo=True)
    SESSION = sessionmaker(bind=ENGINE)()

    Dataset.metadata.create_all(ENGINE)
    File.metadata.create_all(ENGINE)


def teardown():
    SESSION.close()
    ENGINE.dispose()


def test_add_file():
    dataset = Dataset(name='test',
                      base_path='test',
                      filesystem_wrapper_type='test',
                      file_wrapper_type='test',
                      description='test',
                      version='test')
    SESSION.add(dataset)
    SESSION.commit()

    now = datetime.datetime.now()
    file = File(dataset=dataset, path='test', created_at=now, updated_at=now)
    SESSION.add(file)
    SESSION.commit()

    assert SESSION.query(File).filter(File.path == 'test').first() is not None
    assert SESSION.query(File).filter(File.path == 'test').first().dataset == dataset
    assert SESSION.query(File).filter(File.path == 'test').first().created_at == now
    assert SESSION.query(File).filter(File.path == 'test').first().updated_at == now


def test_update_file():
    dataset = Dataset(name='test',
                      base_path='test',
                      filesystem_wrapper_type='test',
                      file_wrapper_type='test',
                      description='test',
                      version='test')
    SESSION.add(dataset)
    SESSION.commit()

    now = datetime.datetime.now()
    file = File(dataset=dataset, path='test', created_at=now, updated_at=now)
    SESSION.add(file)
    SESSION.commit()

    now = datetime.datetime.now()

    SESSION.query(File).filter(File.path == 'test').update({
        'path': 'test2',
        'created_at': now,
        'updated_at': now
    })
    SESSION.commit()

    assert SESSION.query(File).filter(File.path == 'test2').first() is not None
    assert SESSION.query(File).filter(File.path == 'test2').first().dataset == dataset
    assert SESSION.query(File).filter(File.path == 'test2').first().created_at == now
    assert SESSION.query(File).filter(File.path == 'test2').first().updated_at == now

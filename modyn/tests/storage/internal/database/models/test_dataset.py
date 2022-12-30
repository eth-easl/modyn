from modyn.storage.internal.database.models.dataset import Dataset

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


SESSION = None
ENGINE = None


def setup():
    global ENGINE, SESSION  # pylint: disable=global-statement # noqa: E262
    ENGINE = create_engine('sqlite:///:memory:', echo=True)
    SESSION = sessionmaker(bind=ENGINE)()

    Dataset.metadata.create_all(ENGINE)


def teardown():
    SESSION.close()
    ENGINE.dispose()


def test_add_dataset():
    dataset = Dataset(name='test',
                      base_path='test',
                      filesystem_wrapper_type='test',
                      file_wrapper_type='test',
                      description='test',
                      version='test')
    SESSION.add(dataset)
    SESSION.commit()

    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first() is not None
    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first().base_path == 'test'
    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first().filesystem_wrapper_type == 'test'
    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first().file_wrapper_type == 'test'
    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first().description == 'test'
    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first().version == 'test'


def test_update_dataset():
    dataset = Dataset(name='test',
                      base_path='test',
                      filesystem_wrapper_type='test',
                      file_wrapper_type='test',
                      description='test',
                      version='test')
    SESSION.add(dataset)
    SESSION.commit()

    SESSION.query(Dataset).filter(Dataset.name == 'test').update({
        'base_path': 'test2',
        'filesystem_wrapper_type': 'test2',
        'file_wrapper_type': 'test2',
        'description': 'test2',
        'version': 'test2'
    })
    SESSION.commit()

    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first().base_path == 'test2'
    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first().filesystem_wrapper_type == 'test2'
    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first().file_wrapper_type == 'test2'
    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first().description == 'test2'
    assert SESSION.query(Dataset).filter(Dataset.name == 'test').first().version == 'test2'

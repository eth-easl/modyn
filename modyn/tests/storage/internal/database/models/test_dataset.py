from modyn.storage.internal.database.models.dataset import Dataset

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def setup():
    global engine, session
    engine = create_engine('sqlite:///:memory:', echo=True)
    session = sessionmaker(bind=engine)()

    Dataset.metadata.create_all(engine)


def teardown():
    session.close()
    engine.dispose()


def test_add_dataset():
    dataset = Dataset(name='test', base_path='test', filesystem_wrapper_type='test', file_wrapper_type='test', description='test', version='test')
    session.add(dataset)
    session.commit()

    assert session.query(Dataset).filter(Dataset.name == 'test').first() is not None
    assert session.query(Dataset).filter(Dataset.name == 'test').first().base_path == 'test'
    assert session.query(Dataset).filter(Dataset.name == 'test').first().filesystem_wrapper_type == 'test'
    assert session.query(Dataset).filter(Dataset.name == 'test').first().file_wrapper_type == 'test'
    assert session.query(Dataset).filter(Dataset.name == 'test').first().description == 'test'
    assert session.query(Dataset).filter(Dataset.name == 'test').first().version == 'test'


def test_update_dataset():
    dataset = Dataset(name='test', base_path='test', filesystem_wrapper_type='test', file_wrapper_type='test', description='test', version='test')
    session.add(dataset)
    session.commit()

    session.query(Dataset).filter(Dataset.name == 'test').update({
        'base_path': 'test2',
        'filesystem_wrapper_type': 'test2',
        'file_wrapper_type': 'test2',
        'description': 'test2',
        'version': 'test2'
    })
    session.commit()

    assert session.query(Dataset).filter(Dataset.name == 'test').first().base_path == 'test2'
    assert session.query(Dataset).filter(Dataset.name == 'test').first().filesystem_wrapper_type == 'test2'
    assert session.query(Dataset).filter(Dataset.name == 'test').first().file_wrapper_type == 'test2'
    assert session.query(Dataset).filter(Dataset.name == 'test').first().description == 'test2'
    assert session.query(Dataset).filter(Dataset.name == 'test').first().version == 'test2'


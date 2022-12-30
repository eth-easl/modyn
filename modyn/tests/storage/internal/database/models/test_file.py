import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.dataset import Dataset

def setup():
    global engine, session
    engine = create_engine('sqlite:///:memory:', echo=True)
    session = sessionmaker(bind=engine)()

    Dataset.metadata.create_all(engine)
    File.metadata.create_all(engine)


def teardown():
    session.close()
    engine.dispose()


def test_add_file():
    dataset = Dataset(name='test', path='test', dataset_wrapper_type='test', description='test', version='test')
    session.add(dataset)
    session.commit()

    now = datetime.datetime.now()
    file = File(dataset=dataset, path='test', created_at=now, updated_at=now)
    session.add(file)
    session.commit()

    assert session.query(File).filter(File.path == 'test').first() is not None
    assert session.query(File).filter(File.path == 'test').first().dataset == dataset
    assert session.query(File).filter(File.path == 'test').first().created_at == now
    assert session.query(File).filter(File.path == 'test').first().updated_at == now


def test_update_file():
    dataset = Dataset(name='test', path='test', dataset_wrapper_type='test', description='test', version='test')
    session.add(dataset)
    session.commit()

    now = datetime.datetime.now()
    file = File(dataset=dataset, path='test', created_at=now, updated_at=now)
    session.add(file)
    session.commit()

    now = datetime.datetime.now()

    session.query(File).filter(File.path == 'test').update({
        'path': 'test2',
        'created_at': now,
        'updated_at': now
    })
    session.commit()

    assert session.query(File).filter(File.path == 'test2').first() is not None
    assert session.query(File).filter(File.path == 'test2').first().dataset == dataset
    assert session.query(File).filter(File.path == 'test2').first().created_at == now
    assert session.query(File).filter(File.path == 'test2').first().updated_at == now

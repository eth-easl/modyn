import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modyn.storage.internal.database.models.sample import Sample
from modyn.storage.internal.database.models.file import File
from modyn.storage.internal.database.models.dataset import Dataset


def setup():
    global engine, session
    engine = create_engine('sqlite:///:memory:', echo=True)
    session = sessionmaker(bind=engine)()

    Dataset.metadata.create_all(engine)
    File.metadata.create_all(engine)
    Sample.metadata.create_all(engine)


def teardown():
    session.close()
    engine.dispose()


def test_add_sample():
    dataset = Dataset(name='test', path='test', dataset_wrapper_type='test', description='test', version='test')
    session.add(dataset)
    session.commit()

    now = datetime.datetime.now()
    file = File(dataset=dataset, path='test', created_at=now, updated_at=now)
    session.add(file)
    session.commit()

    sample = Sample(file=file, external_key='test', index=0)
    session.add(sample)
    session.commit()

    assert session.query(Sample).filter(Sample.external_key == 'test').first() is not None
    assert session.query(Sample).filter(Sample.external_key == 'test').first().file == file
    assert session.query(Sample).filter(Sample.external_key == 'test').first().index == 0


def test_update_sample():
    dataset = Dataset(name='test', path='test', dataset_wrapper_type='test', description='test', version='test')
    session.add(dataset)
    session.commit()

    now = datetime.datetime.now()
    file = File(dataset=dataset, path='test', created_at=now, updated_at=now)
    session.add(file)
    session.commit()

    sample = Sample(file=file, external_key='test', index=0)
    session.add(sample)
    session.commit()

    session.query(Sample).filter(Sample.external_key == 'test').update({
        'index': 1
    })

    assert session.query(Sample).filter(Sample.external_key == 'test').first().index == 1